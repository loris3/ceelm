from time import time
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pickle, os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

import gc

class IFEngineGeneration(object):
    '''
    This class computes the influence function for every validation data point
    '''
    def __init__(self):
        self.time_dict = defaultdict(list)
        self.hvp_dict = defaultdict(list)
        self.IF_dict = defaultdict(list)
        

    def preprocess_gradients(self, test_grad_dict, get_gradient):
        
        self.test_grad_dict = test_grad_dict
        self.get_gradient = get_gradient
      
        self.n_val = len(self.test_grad_dict.keys())
        self.compute_val_grad_avg()

    def compute_hvps(self, tokenized_train_dataset, train_dataset_name, train_dataset_split, gradient_out_dir, lambda_const_param=10):
        self.compute_hvp_proposed(tokenized_train_dataset, train_dataset_name, train_dataset_split, gradient_out_dir, lambda_const_param=lambda_const_param)
   
    
    def compute_val_grad_avg(self):
        # Compute the avg gradient on the validation dataset
        print("compute_val_grad_avg start",flush=True)
        self.val_grad_avg_dict={}
    
        for weight_name in self.test_grad_dict[0]:
            self.val_grad_avg_dict[weight_name]=torch.zeros(self.test_grad_dict[0][weight_name].shape)
            for val_id in self.test_grad_dict:
                self.val_grad_avg_dict[weight_name] += self.test_grad_dict[val_id][weight_name] / self.n_val
        print("compute_val_grad_avg end",flush=True)
    def compute_hvp_proposed(
        self,
        tokenized_train_dataset,
        train_dataset_name,
        train_dataset_split,
        gradient_out_dir,
        lambda_const_param=10,
        batch_size=128,
        
    ):
        start_time = time()
        os.makedirs(os.path.join(gradient_out_dir), exist_ok=True)
        hvp_cache_path = os.path.join(gradient_out_dir, "hvp.pt")
        hvp_proposed_dict = defaultdict(dict)
        if os.path.exists(hvp_cache_path):
            print(f"Loading hvp_dict from {hvp_cache_path}", flush=True)
            with open(hvp_cache_path, "rb") as f:
                hvp_proposed_dict = pickle.load(f)
        else:
            
            n_train = len(tokenized_train_dataset)
            n_val = len(self.test_grad_dict)

            val_grad_flat_dict = {
                val_id: {name: grad.view(1, -1) for name, grad in self.test_grad_dict[val_id].items()}
                for val_id in self.test_grad_dict
            }

            # pre-compute layer-wise lambda 
            S_sum_dict = {name: 0.0 for name in self.test_grad_dict[0]}
            for i in tqdm(range(0, n_train, batch_size), desc="Lambda pass (1/3)"):
                batch_ids = list(range(i, min(i + batch_size, n_train)))
                batch_grads_list = self.get_gradient(
                    tokenized_train_dataset, train_dataset_name, train_dataset_split, batch_ids
                )
                for weight_name in S_sum_dict:
                    batch_grads = torch.stack([grad[weight_name] for grad in batch_grads_list])
                    S_sum_dict[weight_name] += batch_grads.pow(2).mean(dim=tuple(range(1, batch_grads.ndim))).sum()

            lambda_dict = {name: (S_sum / n_train) / lambda_const_param for name, S_sum in S_sum_dict.items()}

            # compute HVPs for all validation examples batched
            hvp_accumulators = {
                val_id: {name: torch.zeros_like(self.test_grad_dict[val_id][name])
                        for name in self.test_grad_dict[val_id]}
                for val_id in self.test_grad_dict
            }

            for i in tqdm(range(0, n_train, batch_size), desc="HVP pass (2/3)"):
                batch_ids = list(range(i, min(i + batch_size, n_train)))
                batch_grads_list = self.get_gradient(
                    tokenized_train_dataset, train_dataset_name, train_dataset_split, batch_ids
                )

                for weight_name in self.test_grad_dict[0]:
                    batch_grads = torch.stack([grad[weight_name] for grad in batch_grads_list])
                    batch_flat = batch_grads.view(batch_grads.shape[0], -1)

                    for val_id in self.test_grad_dict:
                        val_grad_flat = val_grad_flat_dict[val_id][weight_name]
                        dot = (batch_flat * val_grad_flat).sum(dim=1)
                        denom = lambda_dict[weight_name] + (batch_flat ** 2).sum(dim=1)
                        C = dot / denom
                        hvp_accumulators[val_id][weight_name] += ((val_grad_flat - C[:, None] * batch_flat).sum(dim=0)).view(
                            self.test_grad_dict[val_id][weight_name].shape
                        )

            for val_id in self.test_grad_dict:
                for weight_name in self.test_grad_dict[val_id]:
                    hvp_accumulators[val_id][weight_name] /= (n_train * lambda_dict[weight_name])
                    hvp_proposed_dict[val_id][weight_name] = hvp_accumulators[val_id][weight_name]

            with open(hvp_cache_path, "wb") as f:
                pickle.dump(hvp_proposed_dict, f)

        self.hvp_dict["proposed"] = hvp_proposed_dict
        self.time_dict["proposed"] = time() - start_time

    # def compute_hvp_proposed(self, tokenized_train_dataset, train_dataset_name, train_dataset_split, lambda_const_param=10, batch_size=128):
    #     start_time = time()
    #     hvp_proposed_dict = {}
    #     n_train = len(tokenized_train_dataset)

    #     for weight_name in tqdm(self.val_grad_avg_dict, desc="Weights"):
    #         val_grad_avg = self.val_grad_avg_dict[weight_name]
    #         hvp = torch.zeros_like(val_grad_avg)


    #         S_sum = 0.0
    #         for i in range(0, n_train, batch_size):
    #             batch_ids = list(range(i, min(i + batch_size, n_train)))

    #             batch_grads = torch.stack([gradient[weight_name] for gradient in self.get_gradient(tokenized_train_dataset, train_dataset_name, train_dataset_split, batch_ids)])

    #             S_sum += batch_grads.pow(2).mean(dim=tuple(range(1, batch_grads.ndim))).sum()
    #         lambda_const = (S_sum / n_train) / lambda_const_param

     
    #         for i in range(0, n_train, batch_size):
    #             batch_ids = list(range(i, min(i + batch_size, n_train)))
    #             batch_grads = torch.stack([gradient[weight_name] for gradient in self.get_gradient(tokenized_train_dataset, train_dataset_name, train_dataset_split, batch_ids)])

          
    #             val_grad_flat = val_grad_avg.view(1, -1)
    #             batch_flat = batch_grads.view(batch_grads.shape[0], -1)
    #             dot = (batch_flat * val_grad_flat).sum(dim=1)
    #             denom = lambda_const + (batch_flat ** 2).sum(dim=1)
    #             C = dot / denom

    #             hvp += ((val_grad_flat - C[:, None] * batch_flat).sum(dim=0)).view(val_grad_avg.shape)

    #         hvp /= (n_train * lambda_const)
    #         hvp_proposed_dict[weight_name] = hvp

    #     self.hvp_dict['proposed'] = hvp_proposed_dict
    #     self.time_dict['proposed'] = time() - start_time
    # def compute_hvp_proposed(self, tokenized_train_dataset, train_dataset_name, train_dataset_split, lambda_const_param=10, batch_size=64):
    #     start_time = time()
    #     hvp_proposed_dict = {name: torch.zeros_like(val) for name, val in self.val_grad_avg_dict.items()}
    #     n_train = len(tokenized_train_dataset)

 
    #     val_grad_flat_dict = {name: val.view(1, -1) for name, val in self.val_grad_avg_dict.items()}

    #     S_sum_dict = {name: 0.0 for name in self.val_grad_avg_dict}

    #     # one (batched) pass over the dataset to get lambda
    #     for i in tqdm(range(0, n_train, batch_size), desc="Pass 1/2"):
    #         batch_ids = list(range(i, min(i + batch_size, n_train)))
    #         batch_grads_list = self.get_gradient(tokenized_train_dataset, train_dataset_name, train_dataset_split, batch_ids)
    #         for weight_name in self.val_grad_avg_dict:
    #             val_grad_flat = val_grad_flat_dict[weight_name]

    #             batch_grads = torch.stack([grad[weight_name] for grad in batch_grads_list])
    #             batch_flat = batch_grads.view(batch_grads.shape[0], -1)
    #             S_sum_dict[weight_name] += batch_grads.pow(2).mean(dim=tuple(range(1, batch_grads.ndim))).sum()


    #     lambda_dict = {name: (S_sum / n_train) / lambda_const_param for name, S_sum in S_sum_dict.items()}

    #     # one more (batched) pass to get influence scores
    #     for i in tqdm(range(0, n_train, batch_size), desc="Pass 2/2"):
    #         batch_ids = list(range(i, min(i + batch_size, n_train)))
    #         batch_grads_list = self.get_gradient(tokenized_train_dataset, train_dataset_name, train_dataset_split, batch_ids)

    #         for weight_name in self.val_grad_avg_dict:
    #             val_grad_flat = val_grad_flat_dict[weight_name]
    #             batch_grads = torch.stack([grad[weight_name] for grad in batch_grads_list])
    #             batch_flat = batch_grads.view(batch_grads.shape[0], -1)

    #             dot = (batch_flat * val_grad_flat).sum(dim=1)
    #             denom = lambda_dict[weight_name] + (batch_flat ** 2).sum(dim=1)
    #             C = dot / denom

    #             hvp_proposed_dict[weight_name] += ((val_grad_flat - C[:, None] * batch_flat).sum(dim=0)).view(self.val_grad_avg_dict[weight_name].shape)

    #     # normalize
    #     for weight_name in hvp_proposed_dict:
    #         hvp_proposed_dict[weight_name] /= (n_train * lambda_dict[weight_name])

    #     self.hvp_dict['proposed'] = hvp_proposed_dict
    #     self.time_dict['proposed'] = time() - start_time

    # def compute_IF(self, tokenized_train_dataset, train_dataset_name, train_dataset_split):
    #     for method_name in self.hvp_dict:
    #         if_tmp_dict = {}
    #         for tr_id in tqdm(range(len(tokenized_train_dataset)), desc="Summing influence scores"):
    #             if_tmp_value = 0
    #             for weight_name in self.val_grad_avg_dict:
    #                 if_tmp_value += torch.sum(self.hvp_dict[method_name][weight_name]*self.get_gradient(tokenized_train_dataset, train_dataset_name, train_dataset_split, tr_id)[weight_name])
    #             if_tmp_dict[tr_id]= -if_tmp_value 
                
    #         self.IF_dict[method_name] = pd.DataFrame(if_tmp_dict, index=list(range(len(tokenized_train_dataset))), columns=list(self.test_grad_dict.keys()), dtype=float).T   
    # def compute_IF(self, tokenized_train_dataset, train_dataset_name, train_dataset_split):
    #     for method_name in self.hvp_dict:
    #         print("Computing IF for method: ", method_name)
    #         if_tmp_dict = defaultdict(dict)

    
    #         for tr_id in tqdm(range(len(tokenized_train_dataset)), desc="Summing influence scores"):
    #             tr_gradient = self.get_gradient(
    #                 tokenized_train_dataset, train_dataset_name, train_dataset_split, tr_id
    #             )

         
    #             for val_id in self.test_grad_dict:
    #                 if_tmp_value = 0
    #                 for weight_name in self.test_grad_dict[val_id]:
    #                     if_tmp_value += torch.sum(
    #                         self.hvp_dict[method_name][val_id][weight_name] *
    #                         tr_gradient[weight_name]
    #                     )
    #                 if_tmp_dict[tr_id][val_id] = -if_tmp_value

          
    #         self.IF_dict[method_name] = pd.DataFrame(if_tmp_dict, dtype=float)

    # def compute_IF(self, tokenized_train_dataset, train_dataset_name, train_dataset_split):
    #     for method_name in self.hvp_dict:
    #         print("Computing IF for method: ", method_name)
    #         if_tmp_dict = defaultdict(dict)
    #         for tr_id in tqdm(range(len(tokenized_train_dataset)), desc="Summing influence scores"):
    #             train_grad = self.get_gradient(tokenized_train_dataset, train_dataset_name, train_dataset_split, tr_id)
    #             for val_id in self.test_grad_dict:
    #                 if_tmp_value = 0
    #                 for weight_name in self.test_grad_dict[0]:
    #                     if_tmp_value += torch.sum(self.hvp_dict[method_name][val_id][weight_name]*train_grad[weight_name])
    #                 if_tmp_dict[tr_id][val_id]=-if_tmp_value

    #         self.IF_dict[method_name] = pd.DataFrame(if_tmp_dict, dtype=float)   
    def compute_IF(
        self,
        tokenized_train_dataset,
        train_dataset_name,
        train_dataset_split,
        batch_size=4096
    ):
        start_time = time()
        for method_name in self.hvp_dict:
            print("Computing IF for method:", method_name)

            n_train = len(tokenized_train_dataset)
            n_val = len(self.test_grad_dict)
            weight_names = list(self.test_grad_dict[0].keys())

            val_flat_dict = {
                val_id: torch.cat([self.hvp_dict[method_name][val_id][w].view(-1) for w in weight_names])
                for val_id in self.test_grad_dict
            }

            # n_val x total_weights
            val_matrix = torch.stack([val_flat_dict[val_id] for val_id in self.test_grad_dict])

      
            IF_matrix = torch.zeros(n_train, n_val)

            # loop over training dataset in batches
            for i in tqdm(range(0, n_train, batch_size), desc="Batched IF computation"):
                batch_ids = list(range(i, min(i + batch_size, n_train)))
                batch_grads_list = self.get_gradient(
                    tokenized_train_dataset, train_dataset_name, train_dataset_split, batch_ids
                )

              
                batch_train_matrix = torch.stack([
                    torch.cat([grad[w].view(-1) for w in weight_names])
                    for grad in batch_grads_list
                ])  

                # batch_size x n_val
                if_scores_batch = -batch_train_matrix @ val_matrix.T

             
                IF_matrix[batch_ids] = if_scores_batch.to(IF_matrix.dtype)


                del batch_train_matrix, if_scores_batch
                torch.cuda.empty_cache() 
                gc.collect()
           
            self.IF_dict[method_name] = pd.DataFrame(
                IF_matrix.numpy().T,
                columns=[d["indices"] for d in tokenized_train_dataset],  
                index=list(self.test_grad_dict.keys()),                
                dtype=float
            )
        self.time_dict["IF"] = time() - start_time



    def save_result(self, run_id=0):
        results={}
        results['runtime']=self.time_dict
        results['influence']=self.IF_dict

        with open(f"./results_{run_id}.pkl",'wb') as file:
            pickle.dump(results, file)
