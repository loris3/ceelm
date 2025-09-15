from tqdm import tqdm
import pickle
import torch
from torch.optim import AdamW
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model
)
from datasets import Dataset
import evaluate

import os
from trak.projectors import CudaProjector, NoOpProjector
from trak.projectors import ProjectionType

import shutil
class LORAEngineGeneration(object):
    def __init__(self, 
                model,
                model_path,
                param_string,
                # base_path,
                # project_path,
                # dataset_name='math_with_reason',
                device="cuda",proj_dim=2**13):
        # self.base_path = base_path
        # self.project_path = project_path
        # self.adapter_path = f"{self.project_path}/models/math_without_reason_13bf"
        # self.dataset_name = dataset_name
        self.proj_dim = proj_dim
        self.device=device
        self.model = model
        self.model_path = model_path
        self.param_string = param_string
        # set self.grad_dim
        dummy_input = torch.tensor([[0,0,0,0]]).to(self.device)
        self.model.eval()
        self.model.zero_grad()
        outputs = self.model(input_ids=dummy_input, labels=dummy_input)
        loss = outputs.loss
        loss.backward()
        
        
        # we divide the total projection budget (self.proj_dim) among LoRA parameters proportional to their gradient size
        # while ensuring multiple of 512 for TRAK kernel
        self.param_dims = {}
        self.total_grad_dim = 0

        # Collect gradient dimensions
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            if 'lora_A' in name:
                dim = grad.shape[-1]
            elif 'lora_B' in name:
                dim = grad.T.shape[-1] if grad.ndim >= 2 else grad.shape[-1]
            else:
                continue
            self.param_dims[name] = dim
            self.total_grad_dim += dim

        self.param_proj_dim = {}
        self.param_projectors = {}


        for name, dim in self.param_dims.items():

            proj_dim = min(dim, int(self.proj_dim * dim / self.total_grad_dim))
            
            if proj_dim <= 512:
                self.param_projectors[name] = NoOpProjector()
                proj_dim = dim  # keep original dimension for total sum
            else:
                proj_dim = max(512, round(proj_dim / 512) * 512)
                self.param_projectors[name] = CudaProjector(
                    grad_dim=dim,
                    proj_dim=proj_dim,
                    seed=42,
                    proj_type=ProjectionType.rademacher,
                    device=self.device,
                    max_batch_size=8
                )

            self.param_proj_dim[name] = proj_dim

        self.total_grad_dim_proj = sum(self.param_proj_dim.values())

        print("self.proj_dim", self.proj_dim)
        print("self.total_grad_dim_proj", self.total_grad_dim_proj)
        print("self.total_grad_dim", self.total_grad_dim)
        print("self.param_projectors", self.param_proj_dim, flush=True)

        
    def compute_gradient(self, tokenized_dataset, tokenizer, dataset_name, dataset_split_name, gradient_cache_dir):
       
       
       
        print("compute_gradient", flush=True)

        world_size = torch.cuda.device_count()
        batch_size = (len(tokenized_dataset) + world_size - 1) // world_size
        chunks = [
            tokenized_dataset.select(range(i * batch_size, min((i + 1) * batch_size, len(tokenized_dataset))))
            for i in range(world_size)
        ]
        
        fn = partial(
            batch_map,
     
            tokenizer=tokenizer,
            model=self.model,
            param_projectors=self.param_projectors,
            dataset_name=dataset_name,
            dataset_split_name=dataset_split_name,
            gradient_cache_dir=gradient_cache_dir
        )
        

        
        with ProcessPoolExecutor(max_workers=world_size) as executor:
            futures = []
            for rank, chunk in enumerate(chunks):
                batch_dict = {k: [ex[k] for ex in chunk] for k in chunk.column_names}
                futures.append(executor.submit(fn, batch_dict, rank))
            for f in futures:
                f.result()

        

    
        
from functools import partial
from ..estimator import store_gradient


def batch_map(batch, rank, tokenizer, model, param_projectors, dataset_name, dataset_split_name, gradient_cache_dir):
    
        if rank is None:
            rank = 0
        batch_list = [{k: v[i] for k, v in batch.items()} for i in range(len(batch["input_ids"]))]
        
        collate_fn = lambda x: tokenizer.pad(x, padding="longest", return_tensors="pt")
        dataloader = DataLoader(batch_list, shuffle=False, collate_fn=collate_fn, batch_size=1)
        for grad_dict in _compute_gradient_batch(dataloader, rank, model, param_projectors):
            store_gradient(gradient_cache_dir, dataset_name, dataset_split=dataset_split_name, gradient_dict=grad_dict) 
        
        
   
    
    
def _compute_gradient_batch(dataloader, rank, model, param_projectors):
    print("_compute_gradient_batch", flush=True)
    device = f"cuda:{rank % torch.cuda.device_count()}"
    model.to(device)
    model.eval()
    grad_dicts = []

    for row in tqdm(dataloader, position=rank, desc=f"Worker {rank}"):
        model.zero_grad()
        row['labels'] = row['input_ids']
        row.to(device)
        outputs = model(**row)
        loss = outputs.loss
        loss.backward()
        grad_dict = {}
        for k, v in model.named_parameters():
            grad = None
            if k not in param_projectors:
                continue
            if 'lora_A' in k:
                grad = v.grad
               
            elif 'lora_B' in k:
                grad = v.grad.T
            with torch.no_grad():
                grad_dict[k] = param_projectors[k].project(grad.contiguous(), model_id=0).cpu()
                del grad
        grad_dicts.append({row["indices"][0].item(): grad_dict})
    return grad_dicts
