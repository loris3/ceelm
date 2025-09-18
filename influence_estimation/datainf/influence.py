from time import time
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pickle, os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed


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

    def compute_hvps(self, tokenized_train_dataset, train_dataset_name, train_dataset_split, lambda_const_param=10):
        self.compute_hvp_proposed(tokenized_train_dataset, train_dataset_name, train_dataset_split, lambda_const_param=lambda_const_param)
   
    
    def compute_val_grad_avg(self):
        # Compute the avg gradient on the validation dataset
        print("compute_val_grad_avg start",flush=True)
        self.val_grad_avg_dict={}
    
        for weight_name in self.test_grad_dict[0]:
            self.val_grad_avg_dict[weight_name]=torch.zeros(self.test_grad_dict[0][weight_name].shape)
            for val_id in self.test_grad_dict:
                self.val_grad_avg_dict[weight_name] += self.test_grad_dict[val_id][weight_name] / self.n_val
        print("compute_val_grad_avg end",flush=True)
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
    def compute_hvp_proposed(self, tokenized_train_dataset, train_dataset_name, train_dataset_split, lambda_const_param=10, batch_size=64):
        start_time = time()
        hvp_proposed_dict = {name: torch.zeros_like(val) for name, val in self.val_grad_avg_dict.items()}
        n_train = len(tokenized_train_dataset)

 
        val_grad_flat_dict = {name: val.view(1, -1) for name, val in self.val_grad_avg_dict.items()}

        S_sum_dict = {name: 0.0 for name in self.val_grad_avg_dict}

        # one (batched) pass over the dataset to get lambda
        for i in tqdm(range(0, n_train, batch_size), desc="Pass 1/2"):
            batch_ids = list(range(i, min(i + batch_size, n_train)))
            batch_grads_list = self.get_gradient(tokenized_train_dataset, train_dataset_name, train_dataset_split, batch_ids)
            for weight_name in self.val_grad_avg_dict:
                val_grad_flat = val_grad_flat_dict[weight_name]

                batch_grads = torch.stack([grad[weight_name] for grad in batch_grads_list])
                batch_flat = batch_grads.view(batch_grads.shape[0], -1)
                S_sum_dict[weight_name] += batch_grads.pow(2).mean(dim=tuple(range(1, batch_grads.ndim))).sum()


        lambda_dict = {name: (S_sum / n_train) / lambda_const_param for name, S_sum in S_sum_dict.items()}

        # one more (batched) pass to get influence scores
        for i in tqdm(range(0, n_train, batch_size), desc="Pass 2/2"):
            batch_ids = list(range(i, min(i + batch_size, n_train)))
            batch_grads_list = self.get_gradient(tokenized_train_dataset, train_dataset_name, train_dataset_split, batch_ids)

            for weight_name in self.val_grad_avg_dict:
                val_grad_flat = val_grad_flat_dict[weight_name]
                batch_grads = torch.stack([grad[weight_name] for grad in batch_grads_list])
                batch_flat = batch_grads.view(batch_grads.shape[0], -1)

                dot = (batch_flat * val_grad_flat).sum(dim=1)
                denom = lambda_dict[weight_name] + (batch_flat ** 2).sum(dim=1)
                C = dot / denom

                hvp_proposed_dict[weight_name] += ((val_grad_flat - C[:, None] * batch_flat).sum(dim=0)).view(self.val_grad_avg_dict[weight_name].shape)

        # normalize
        for weight_name in hvp_proposed_dict:
            hvp_proposed_dict[weight_name] /= (n_train * lambda_dict[weight_name])

        self.hvp_dict['proposed'] = hvp_proposed_dict
        self.time_dict['proposed'] = time() - start_time

    def compute_IF(self, tokenized_train_dataset, train_dataset_name, train_dataset_split):
        for method_name in self.hvp_dict:
            if_tmp_dict = {}
            for tr_id in tqdm(range(len(tokenized_train_dataset)), desc="Summing influence scores"):
                if_tmp_value = 0
                for weight_name in self.val_grad_avg_dict:
                    if_tmp_value += torch.sum(self.hvp_dict[method_name][weight_name]*self.get_gradient(tokenized_train_dataset, train_dataset_name, train_dataset_split, tr_id)[weight_name])
                if_tmp_dict[tr_id]= -if_tmp_value 
                
            self.IF_dict[method_name] = pd.DataFrame(if_tmp_dict, index=list(range(len(tokenized_train_dataset))), columns=list(self.test_grad_dict.keys()), dtype=float).T   
    
    def save_result(self, run_id=0):
        results={}
        results['runtime']=self.time_dict
        results['influence']=self.IF_dict

        with open(f"./results_{run_id}.pkl",'wb') as file:
            pickle.dump(results, file)
