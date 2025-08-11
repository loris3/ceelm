
import torch

from datasets import load_dataset

import itertools
import numpy as np
from scipy import stats
import itertools
import pandas as pd

from torch.nn.functional import normalize
import os
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType



class Selector():
    def __init__(self,base_model_path, tokenizer, train_dataset, normalize=True, proj_dim=8192, param_list=[]):
        self.base_model_path = base_model_path
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.proj_dim = proj_dim
        self.normalize = normalize
        param_list = [self.normalize, proj_dim, os.path.basename(base_model_path)]
        self.param_string = "-".join([str(s) for s in param_list])
        self.gradient_cache_dir = os.path.join("./cache/gradients", self.__class__.__name__, self.param_string)

        collate_fn = lambda x: self.tokenizer.pad(x, padding="longest", return_tensors="pt")
        
        
        def tokenize(example):
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
            return tokenizer(text, truncation=True, padding="max_length", max_length=2048)
        dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
        
        self.compute_gradient(dataset, collate_fn)
    def _compute_gradient_batch(self, fingerprint, dataloader, rank, save_path, gradient_type="adam"):
        try:


            
            device = f"cuda:{rank % torch.cuda.device_count()}"
            print("rank", rank, device, flush=True)
            model_id = 0  # model_id is used to draft the random seed for the projectors
            block_size = 128  # fixed block size for the projectors
            projector_batch_size = 32  # batch size for the projectors
            torch.random.manual_seed(0)  # set the random seed for torch
            
            model = AutoModelForCausalLM.from_pretrained(self.base_model_path, torch_dtype=torch.float16, load_in_8bit=True, device_map={ "": device })
            # model.to(device)
            model.eval()
    
            
            # if gradient_type == "adam":
            #     assert self.adam_optimizer_state is not None
            #     m, v = self.prepare_optimizer_state()
            number_of_params = get_number_of_params(model)
            projector = CudaProjector(grad_dim=number_of_params,
                            proj_dim=self.proj_dim,
                            seed=0,
                            proj_type=ProjectionType.rademacher,
                            device=device,
                            block_size=block_size,
                            max_batch_size=projector_batch_size)       
            
            count = 0
            
            def _project(current_full_grads, projected_grads):
                current_full_grads = torch.stack(current_full_grads)#.to(torch.bfloat16)
                current_projected_grads = projector.project(
                    current_full_grads, model_id=model_id)
                projected_grads.append(current_projected_grads.cpu())
                
            def obtain_gradients(batch, model, device):
             
                batch['labels'] = batch['input_ids']
                for key in batch:
                    if batch[key] is not None:
                        batch[key] = batch[key].to(device)
                
                try:
                    loss = model(**batch).loss
                except:
                    print("device", device)
                    print("batch", batch,flush=True)
                    # import traceback
                    print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
                    print("ba", batch, flush=True)
                    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                    # traceback.print_exc()
                    
                loss.backward()
                vectorized_grads = torch.cat(
                    [p.grad.view(-1) for p in model.parameters() if p.grad is not None]).to(torch.float16)
                return vectorized_grads

            full_grads = []  
            projected_grads = []  

            for batch in tqdm(dataloader, total=len(dataloader), position=rank, desc=f"Worker {rank}"):
                for key in batch:
                    if batch[key] is not None:
                        batch[key] = batch[key].to(device)
                count += 1
                # if gradient_type == "adam":
                #     vectorized_grads = self.obtain_gradients_with_adam(batch, m, v)
                if True or gradient_type == "sgd":
                    vectorized_grads = obtain_gradients(batch, model, device)

                full_grads.append(vectorized_grads)
                model.zero_grad()

                if count % projector_batch_size == 0:
                    _project(full_grads, projected_grads)
                    full_grads = []
            if len(full_grads) > 0:
                _project(full_grads, projected_grads)
                full_grads = []

            torch.cuda.empty_cache()
            merged_data = torch.cat(projected_grads, dim=0)
            del projected_grads
            if self.normalize:
                
                merged_data = normalize(merged_data, dim=1)

            
        
            torch.save(merged_data, save_path)
        except:
            import traceback
            print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
            traceback.print_exc()
    def compute_gradient(self, tokenized_dataset, collate_fn):
        partial_results_dir = os.path.join("./cache/gradients/partial", tokenized_dataset._fingerprint)
        
        def batch_map(batch, rank):
            if rank is None:
                rank = 0
            batch_list = [{k: v[i] for k, v in batch.items()} for i in range(len(batch["input_ids"]))]
            dataloader = DataLoader(batch_list, shuffle=False, collate_fn=collate_fn, batch_size=1)
            save_path = os.path.join(partial_results_dir, f"grads_rank_{rank}.pt")
            if not os.path.exists(save_path):
                os.makedirs(partial_results_dir, exist_ok=True)
                self._compute_gradient_batch(tokenized_dataset._fingerprint, dataloader, rank, save_path)
            
            return {"_": [None] * len(batch_list)}
        
        
        # we manually write to disk as .map is slow with large objects
        tokenized_dataset.map(
            batch_map,
            batched=True,
            batch_size=100_000,
            with_rank=True,
            num_proc=2,#torch.cuda.device_count(),
        )
        print("done")
        
        
        # grads = []


        # files = [f for f in os.listdir(partial_results_dir) if f.startswith("grads_rank_") and f.endswith(".pt")]
        # files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))  # grads_rank_{rank}.pt
       
        # assert len(files) == torch.cuda.device_count()
        # for fname in files:
        #     path = os.path.join(partial_results_dir, fname)
        #     grad_dict = torch.load(path)
        #     grads.extend(grad_dict)
        #     print("fname",fname)
        # # shutil.rmtree(partial_results_dir) 
        # return {i : grad for i, grad in enumerate(grads)}
def get_number_of_params(model):
    """ Make sure that only lora parameters require gradients in peft models. """
    num_params = sum([p.numel()
                    for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params
if __name__ == "__main__":
    from multiprocess import set_start_method
    set_start_method("spawn")

    base_model_path = "allenai/OLMo-2-0425-1B-SFT"
    
    train_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train").shuffle(seed=0).select(range(200))

    test_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train").shuffle(seed=0).select(range(200))
    


    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    
    selector = Selector(base_model_path, tokenizer, train_dataset)