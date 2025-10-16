from time import time
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pickle, os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

import gc
import psutil
import os
def print_cuda_memory(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        max_reserved = torch.cuda.max_memory_reserved() / (1024**3)
        print(f"[CUDA] {prefix} allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB, "
              f"max_allocated: {max_allocated:.2f} GB, max_reserved: {max_reserved:.2f} GB")

def log_cpu_memory(prefix=""):
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size (actual memory used)
    mem_gb = mem_bytes / (1024**3)
    print(f"[DEBUG] {prefix} CPU memory usage: {mem_gb:.2f} GB")
class IFEngineGeneration(object):
    '''
    This class is a batched version of https://github.com/ykwon0407/DataInf/blob/main/src/influence.py#L136
    It computes the influence of every validation data point.
    Equivalence (up to difference in floating point precision of atol=1e-3) is tested in `datainf_batched_test.py`.
    There are three batch sizes that control how often the training and validation datasets are traversed. 
    '''
    def __init__(self):
        self.time_dict = defaultdict(list)
        self.hvp_dict = defaultdict(list)
        self.IF_dict = defaultdict(list)
        

    def preprocess_gradients(self, val_grad_dict, get_gradient):
        
        self.val_grad_dict = val_grad_dict
        self.get_gradient = get_gradient
      
        self.n_val = len(self.val_grad_dict.keys())
  
    def compute_hvps(self, tokenized_train_dataset, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, gradient_out_dir, lambda_const_param=10):
        self.compute_hvp_proposed(tokenized_train_dataset, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, gradient_out_dir, lambda_const_param=lambda_const_param)
   
    def compute_hvp_proposed(
        self,
        tokenized_train_dataset,
        train_dataset_name,
        train_dataset_split,
        test_dataset_name,
        test_dataset_split,
        gradient_out_dir,
        lambda_const_param=10,
        batch_size=32,
        val_batch_size=1000,
        inner_batch_size=1
    ):
        with torch.no_grad():
            start_time = time()

            print("batch_size", batch_size, "val_batch_size", val_batch_size)
            hvp_cache_path = os.path.join(
                gradient_out_dir,
                "hvp",
                train_dataset_name,
                train_dataset_split,
                test_dataset_name,
                test_dataset_split,
                "hvp.pt",
            )
            os.makedirs(os.path.dirname(hvp_cache_path), exist_ok=True)
            if os.path.exists(hvp_cache_path):
                print(f"Loading cached HVP from {hvp_cache_path}", flush=True)
                with open(hvp_cache_path, "rb") as f:
                    hvp_proposed_dict = pickle.load(f)
                self.hvp_dict["proposed"] = hvp_proposed_dict
                self.time_dict["proposed"] = time() - start_time
                return hvp_proposed_dict
            print("building hvp", flush=True)
            hvp_proposed_dict = defaultdict(dict)
            n_train = len(tokenized_train_dataset)
            val_ids = list(self.val_grad_dict.keys())
            print("val_ids", len(val_ids))

            sample_val_id = next(iter(self.val_grad_dict))
            weight_names = list(self.val_grad_dict[sample_val_id].keys())
            n_weights = len(weight_names)

            print("Computing lambda_const...")
       
            sum_sq = torch.zeros(n_weights, dtype=torch.float64)
            for i in tqdm(range(0, n_train, batch_size), desc="Compute lambda_const"):
                batch_ids = list(range(i, min(i + batch_size, n_train)))
                batch_grads_list = self.get_gradient(train_dataset_name, train_dataset_split, batch_ids)
                for grads_dict in batch_grads_list:
                    per_weight_means = []
                    for w_idx, w in enumerate(weight_names):
                        g = grads_dict[w].view(-1)
                        per_weight_means.append(g.double().pow(2).mean())  
                    sum_sq += torch.tensor(per_weight_means, dtype=torch.float64)
            lambda_const = (sum_sq / n_train) / lambda_const_param
            lambda_const = lambda_const.to(dtype=torch.float32)
                    
                    
                    
                   
      
            print("Computing HVP (single pass over train data per val batch)...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            
            val_ids_list = list(val_ids)
            n_val = len(val_ids_list)
            val_grads_per_weight = []
            
            for w in weight_names:
                per_w = [self.val_grad_dict[v][w].view(-1).float() for v in val_ids_list]
                val_grads_per_weight.append(torch.stack(per_w))  
            
            print_cuda_memory("before loop")
            for vb_start in tqdm(range(0, len(val_ids), val_batch_size), desc="Validation batches"):
                vb_end = min(vb_start + val_batch_size, len(val_ids))
                val_batch = val_ids[vb_start:vb_end]

            
                
                val_grads_batch_per_weight = [
                    vg[vb_start:vb_end].to(device) for vg in val_grads_per_weight
                ] 
                hvp_batch_per_weight = [
                    torch.zeros_like(vg) for vg in val_grads_batch_per_weight
                ]
                lambda_const = lambda_const.to(device)



                for i in tqdm(range(0, n_train, batch_size),desc="Inner train data loop"):
                    batch_ids = list(range(i, min(i + batch_size, n_train)))
                    batch_grads_list = self.get_gradient(train_dataset_name, train_dataset_split, batch_ids)

   

                    batch_grads_per_weight = []
                    for w in weight_names:
                        per_w = [g[w].view(-1).float() for g in batch_grads_list]
                        batch_grads_per_weight.append(torch.stack(per_w).to(device))  # (batch_len, param_len_w)

                    batch_len = len(batch_ids)
                    for k in range(0, batch_len, inner_batch_size):
                        sb_end = min(k + inner_batch_size, batch_len)
                        sb_size = sb_end - k
                        
                        for w_idx in range(n_weights):
                            val_w = val_grads_batch_per_weight[w_idx]           
                            tr_w = batch_grads_per_weight[w_idx][k:sb_end]      

              
                            sub_dot_w = val_w @ tr_w.t()  

                          
                            tr_norm_sq = tr_w.pow(2).sum(dim=1)  # (sb_size,)
                            denom_w = lambda_const[w_idx].view(()) + tr_norm_sq 
                           
                            sub_C_tmp_w = sub_dot_w / denom_w.unsqueeze(0)

                            contrib = (val_w.unsqueeze(1) - sub_C_tmp_w.unsqueeze(2) * tr_w.unsqueeze(0)).sum(dim=1)  

                 
                            denom_factor = (n_train * lambda_const[w_idx]).view(())
                            hvp_batch_per_weight[w_idx] += contrib / denom_factor
                 

                    del batch_grads_per_weight, batch_grads_list
                    torch.cuda.empty_cache()

                vb_size = vb_end - vb_start
                for v_local_idx, val_id in enumerate(val_batch):
                    for w_idx, w in enumerate(weight_names):
                        hvp_proposed_dict[val_id][w] = hvp_batch_per_weight[w_idx][v_local_idx].float().cpu()
            self.hvp_dict["proposed"] = hvp_proposed_dict
            self.time_dict["proposed"] = time() - start_time
            print("got hvp", flush=True)
            with open(hvp_cache_path, "wb") as f:
                pickle.dump(hvp_proposed_dict, f)

            return hvp_proposed_dict

    def compute_IF(self, tokenized_train_dataset, train_dataset_name, train_dataset_split, batch_size: int = 4096):
        tr_ids = tokenized_train_dataset["indices"]
        val_ids = list(self.val_grad_dict.keys())
        weight_names = list(self.val_grad_dict[val_ids[0]].keys())

        n_train = len(tr_ids)
        n_val = len(val_ids)
        if_matrix = torch.zeros(n_train, n_val, dtype=torch.float32)

   
        hvp_flat_dict = {
            val_id: torch.cat([self.hvp_dict["proposed"][val_id][w].view(-1) for w in weight_names]).float()
            for val_id in val_ids
        }


        for i in tqdm(range(0, n_train, batch_size), desc="Train batches"):
            batch_ids = tr_ids[i:i + batch_size]
            batch_grads_list = self.get_gradient(
                train_dataset_name, train_dataset_split, batch_ids
            )
            batch_grads = torch.stack([
                torch.cat([grad[w].view(-1) for w in weight_names]).float()
                for grad in batch_grads_list
            ])  

          
            for v_idx, val_id in enumerate(val_ids):
                hvp_flat = hvp_flat_dict[val_id]
                if_batch = -torch.matmul(batch_grads, hvp_flat)
                if_matrix[i:i + len(batch_ids), v_idx] = if_batch.detach()

            del batch_grads, batch_grads_list
            torch.cuda.empty_cache()

        self.IF_dict["proposed"] = pd.DataFrame(
            if_matrix, index=tr_ids, columns=val_ids, dtype=float
        ).T



    def save_result(self, run_id=0):
        results={}
        results['runtime']=self.time_dict
        results['influence']=self.IF_dict

        with open(f"./results_{run_id}.pkl",'wb') as file:
            pickle.dump(results, file)
