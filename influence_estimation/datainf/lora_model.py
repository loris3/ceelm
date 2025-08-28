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
                device="cuda",proj_dim=512):
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
        
        for k, v in self.model.named_parameters():
            if 'lora_A' in k and v.grad is not None:
                self.grad_dim = v.grad.cpu().shape[-1]
                break
            elif 'lora_B' in k and v.grad is not None:
                self.grad_dim = v.grad.cpu().T.shape[-1]
                break
        print("self.grad_dim",self.grad_dim)
        
        if self.grad_dim <= self.proj_dim:
            self.projector = NoOpProjector()
        else:
            self.projector = CudaProjector(grad_dim=self.grad_dim, proj_dim=self.proj_dim,seed=42, proj_type=ProjectionType.rademacher,device=self.device, max_batch_size=8)
        # self.projector = NoOpProjector()
        # self.load_pretrained_network()
        # self.load_datasets()

    # def load_pretrained_network(self):
    #     # setup tokenizer
    #     self.tokenizer = LlamaTokenizer.from_pretrained(self.base_path)
    #     self.tokenizer.padding_side = "right"
    #     self.tokenizer.pad_token = self.tokenizer.eos_token
    #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    #     # load a base model
    #     quantization_config = BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False)
    #     base_model = LlamaForCausalLM.from_pretrained(
    #         self.base_path,
    #         quantization_config=quantization_config,
    #         torch_dtype=torch.bfloat16,
    #         offload_folder="offload",
    #         offload_state_dict=True,
    #     )

    #     # load a pre-trained model.
    #     self.model = PeftModel.from_pretrained(base_model, self.adapter_path, is_trainable=True)
    #     self.finetuned_config = LoraConfig.from_pretrained(pretrained_model_name_or_path=self.adapter_path)

    # def load_datasets(self):
    #     self.train_dataset = Dataset.load_from_disk(f"{self.project_path}/datasets/{self.dataset_name}_train.hf").shuffle(seed=42).select(range(100))
    #     self.validation_dataset = Dataset.load_from_disk(f"{self.project_path}/datasets/{self.dataset_name}_test.hf").shuffle(seed=42).select(range(100))

    # def create_tokenized_datasets(self):
    #     tokenize_func = lambda x: self.tokenizer(
    #         x["prompt"], truncation=True, padding=True, max_length=128, return_tensors="pt" # text should be more appropritate
    #     ).to(self.device)

    #     if 'with_reason' in self.dataset_name:
    #         column_list=["text", "answer", "variation", "prompt", "reason"]
    #     else:
    #         column_list=["text", "answer", "variation", "prompt"]

    #     tokenized_datasets=dict()
    #     tokenized_datasets["train"] = self.train_dataset.map(
    #         tokenize_func,
    #         batched=True,
    #         remove_columns=column_list,
    #     )
    #     tokenized_datasets["validation"] = self.validation_dataset.map(
    #         tokenize_func,
    #         batched=True,
    #         remove_columns=column_list,
    #     )
    #     collate_fn = lambda x: self.tokenizer.pad(x, padding="longest", return_tensors="pt")

    #     return tokenized_datasets, collate_fn

    def compute_gradient_slow(self, tokenized_dataset, collate_fn):
            
            dataloader = DataLoader(tokenized_dataset, shuffle=False, collate_fn=collate_fn, batch_size=1)
             
            self.model.eval()
            grad_dicts = {}
            
            for step, batch in enumerate(tqdm(dataloader)):
                self.model.zero_grad()
                batch['labels'] = batch['input_ids']
                batch.to(self.device)
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                grad_dict = {}
                for k, v in self.model.named_parameters():
                    if 'lora_A' in k:
                        grad = v.grad
                        with torch.no_grad():
                            grad_dict[k] = self.projector.project(grad.contiguous(), model_id=0).cpu()
                    elif 'lora_B' in k:
                        grad = v.grad.T
                        with torch.no_grad():
                            grad_dict[k] = self.projector.project(grad.contiguous(), model_id=0).cpu()

                grad_dicts[step] = grad_dict
                # del grad_dict
            return grad_dicts

    def _compute_gradient_batch(self, dataloader, rank, partial_results_dir,  dataset_name, dataset_split):
        device = f"cuda:{rank % torch.cuda.device_count()}"
        self.model.to(device)
        self.model.eval()
        grads = []

        for row in tqdm(dataloader, position=rank, desc=f"Worker {rank}"):
            self.model.zero_grad()
            row['labels'] = row['input_ids']
            row.to(device)
            outputs = self.model(**row)
            loss = outputs.loss
            loss.backward()
            grad_dict = {}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad = v.grad
                    with torch.no_grad():
                        grad_dict[k] = self.projector.project(grad, model_id=0).cpu()
                elif 'lora_B' in k:
                    grad = v.grad.T
                    with torch.no_grad():
                        grad_dict[k] = self.projector.project(grad, model_id=0).cpu()
            grads.append(grad_dict)
        print("store",flush=True)


        os.makedirs(partial_results_dir, exist_ok=True)
        save_path = os.path.join(partial_results_dir, f"grads_rank_{rank}.pt")
        torch.save(grads, save_path)
        # return {"_": [None] * len(dataloader)}

    def compute_gradient(self, tokenized_dataset, dataset_name, dataset_split, collate_fn):
       
        partial_results_dir = os.path.join(
            "./cache/gradients/partial",
            self.__class__.__name__,
            "partial",
             self.param_string, os.path.basename(self.model_path),
            "_".join([dataset_name, dataset_split])
        )
        def batch_map(batch, rank):
            if rank is None:
                rank = 0
            batch_list = [{k: v[i] for k, v in batch.items()} for i in range(len(batch["input_ids"]))]
            dataloader = DataLoader(batch_list, shuffle=False, collate_fn=collate_fn, batch_size=1)
            self._compute_gradient_batch(dataloader, rank, partial_results_dir,  dataset_name, dataset_split)
            return {"_": [None] * len(batch_list)}
        
        
        # we manually write to disk as .map is slow with large objects
        tokenized_dataset.map(
            batch_map,
            batched=True,
            with_rank=True,
            batch_size=(len(tokenized_dataset) + torch.cuda.device_count() - 1) // torch.cuda.device_count(),
            num_proc=torch.cuda.device_count(),

        )
        
        
        
        grads = []


        files = [f for f in os.listdir(partial_results_dir) if f.startswith("grads_rank_") and f.endswith(".pt")]
        files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))  # grads_rank_{rank}.pt
       
        assert len(files) == torch.cuda.device_count()
        for fname in files:
            path = os.path.join(partial_results_dir, fname)
            grad_dict = torch.load(path)
            grads.extend(grad_dict)
            print("fname",fname)
        # shutil.rmtree(partial_results_dir) 
        return {i : grad for i, grad in enumerate(grads)}