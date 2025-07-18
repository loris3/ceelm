from time import time
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pickle, os
import torch

class IFEngineGeneration(object):
    '''
    This class computes the influence function for every validation data point
    '''
    def __init__(self):
        self.time_dict = defaultdict(list)
        self.hvp_dict = defaultdict(list)
        self.IF_dict = defaultdict(list)

    def preprocess_gradients(self, tr_grad_dict, val_grad_dict):
        self.tr_grad_dict = tr_grad_dict
        self.val_grad_dict = val_grad_dict

        self.n_train = len(self.tr_grad_dict.keys())
        self.n_val = len(self.val_grad_dict.keys())

    def compute_hvps(self, lambda_const_param=10):
        self.compute_hvp_identity()
        self.compute_hvp_proposed(lambda_const_param=lambda_const_param)
    def compute_hvps_slow(self, lambda_const_param=10):
        self.compute_hvp_identity()
        self.compute_hvp_proposed_slow(lambda_const_param=lambda_const_param)
        # output_new = self.hvp_dict['proposed']
        # self.compute_hvp_proposed_slow(lambda_const_param=lambda_const_param)
        # output_old = self.hvp_dict['proposed']
        # for val_id in output_new:
        #     for weight_name in output_new[val_id]:
        #         diff = torch.abs(output_new[val_id][weight_name] - output_old[val_id][weight_name])
        #         print(f"Max diff for val_id={val_id}, weight_name={weight_name}: {diff.max()}")
        #         assert torch.allclose(
        #             output_new[val_id][weight_name], 
        #             output_old[val_id][weight_name],rtol=1e-3, atol=1e-4
        #         ), f"Mismatch at val_id={val_id}, weight_name={weight_name}"

    def compute_hvp_identity(self):
        start_time = time()
        self.hvp_dict["identity"] = self.val_grad_dict.copy()
        self.time_dict["identity"] = time() - start_time

    def compute_hvp_proposed_slow(self, lambda_const_param=10):
        start_time = time()
        hvp_proposed_dict=defaultdict(dict)
        for val_id in tqdm(self.val_grad_dict.keys()):
            for weight_name in self.val_grad_dict[val_id]:
                # lambda_const computation
                S=torch.zeros(len(self.tr_grad_dict.keys()))
                for tr_id in self.tr_grad_dict:
                    tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                    S[tr_id]=torch.mean(tmp_grad**2)
                lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

                # hvp computation
                hvp=torch.zeros(self.val_grad_dict[val_id][weight_name].shape)
                for tr_id in self.tr_grad_dict:
                    tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                    C_tmp = torch.sum(self.val_grad_dict[val_id][weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
                    hvp += (self.val_grad_dict[val_id][weight_name] - C_tmp*tmp_grad) / (self.n_train*lambda_const)
                hvp_proposed_dict[val_id][weight_name] = hvp
        self.hvp_dict['proposed'] = hvp_proposed_dict
        self.time_dict['proposed'] = time()-start_time
    def compute_hvp_proposed(self, lambda_const_param=10):
        start_time = time()
        hvp_proposed_dict = defaultdict(dict)
        
        tr_ids = list(self.tr_grad_dict.keys())
        n_train = self.n_train
        

        tr_grads_stacked = {}
        for weight_name in next(iter(self.tr_grad_dict.values())).keys():
            grads = [self.tr_grad_dict[tr_id][weight_name].flatten() for tr_id in tr_ids]
            tr_grads_stacked[weight_name] = torch.stack(grads)  # shape (n_train, grad_dim)
        
        for val_id in tqdm(self.val_grad_dict.keys()):
            for weight_name in self.val_grad_dict[val_id]:
                val_grad = self.val_grad_dict[val_id][weight_name].flatten()  # shape (grad_dim,)
             
                S = torch.mean(tr_grads_stacked[weight_name] ** 2, dim=1)  # (n_train,)
                lambda_const = torch.mean(S) / lambda_const_param  # scalar
                
             
                dot_products = torch.matmul(tr_grads_stacked[weight_name], val_grad)  # (n_train,)
                denom = lambda_const + torch.sum(tr_grads_stacked[weight_name] ** 2, dim=1)  # (n_train,)
                C_tmp = dot_products / denom  # (n_train,)
                
              
                weighted_terms = (val_grad.unsqueeze(0) - (C_tmp.unsqueeze(1) * tr_grads_stacked[weight_name]))  # (n_train, grad_dim)
                hvp = weighted_terms.sum(dim=0) / (n_train * lambda_const)  # (grad_dim,)
                
           
                hvp = hvp.view(self.val_grad_dict[val_id][weight_name].shape)
                hvp_proposed_dict[val_id][weight_name] = hvp
        
        self.hvp_dict['proposed'] = hvp_proposed_dict
        self.time_dict['proposed'] = time() - start_time


    def compute_IF_slow(self):
        for method_name in self.hvp_dict:
            print("Computing IF for method (slow): ", method_name)
            if_tmp_dict = defaultdict(dict)
            for tr_id in self.tr_grad_dict:
                for val_id in self.val_grad_dict:
                    if_tmp_value = 0
                    for weight_name in self.val_grad_dict[0]:
                        if_tmp_value += torch.sum(self.hvp_dict[method_name][val_id][weight_name]*self.tr_grad_dict[tr_id][weight_name])
                    if_tmp_dict[tr_id][val_id]=-if_tmp_value

            self.IF_dict[method_name] = pd.DataFrame(if_tmp_dict, dtype=float)   
    def compute_IF(self):
        self.IF_dict = {}
        for method_name in self.hvp_dict:
            print(f"Computing IF (fast) for method: {method_name}")
            tr_ids = list(self.tr_grad_dict.keys())
            val_ids = list(self.val_grad_dict.keys())
            
            tr_grads_stacked = {}
            for weight_name in next(iter(self.tr_grad_dict.values())).keys():
                grads = [self.tr_grad_dict[tr_id][weight_name].flatten().to(torch.float64) for tr_id in tr_ids]
                tr_grads_stacked[weight_name] = torch.stack(grads)
            
            hvp_stacked = {}
            for weight_name in next(iter(self.val_grad_dict.values())).keys():
                grads = [self.hvp_dict[method_name][val_id][weight_name].flatten().to(torch.float64) for val_id in val_ids]
                hvp_stacked[weight_name] = torch.stack(grads)
            
            IF_matrix = torch.zeros(len(tr_ids), len(val_ids), dtype=torch.float64)
            for weight_name in hvp_stacked:
                dot_products = torch.matmul(tr_grads_stacked[weight_name], hvp_stacked[weight_name].T)
                IF_matrix += dot_products
            
            IF_matrix = -IF_matrix
            self.IF_dict[method_name] = pd.DataFrame(IF_matrix.numpy(), index=tr_ids, columns=val_ids).T
    def save_result(self, run_id=0):
        results={}
        results['runtime']=self.time_dict
        results['influence']=self.IF_dict

        with open(f"./results_{run_id}.pkl",'wb') as file:
            pickle.dump(results, file)
