from abc import ABC, abstractmethod
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
import os
import logging
import torch
import pandas as pd
import json 
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class BaseEstimator(ABC):
    def __init__(self, model_path, train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, device="cuda", param_list=[], persistent_cache=False):
        self.model_path = model_path
     
        self.train_dataset_split = train_dataset_split
        self.test_dataset_split = test_dataset_split
        self.train_dataset_name = os.path.basename(train_dataset_name)
        self.test_dataset_name = os.path.basename(test_dataset_name)
       
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset     
        self.device = device
        self.param_string = "-".join([str(s) for s in param_list])
        
        self.gradient_cache = {}
        
        self.gradient_cache_dir = os.path.join("cache/gradients/full" if persistent_cache else "/tmp/cache/gradients/full", self.__class__.__name__, self.param_string, os.path.basename(self.model_path))

        os.makedirs(self.gradient_cache_dir, exist_ok=True)
        
        self.influence_estimate_path = os.path.join(
                "./results/influence",
                self.__class__.__name__,
                self.param_string,
                os.path.basename(self.model_path),
                "_".join([self.train_dataset_name, self.train_dataset_split]),
                "_".join([self.test_dataset_name, self.test_dataset_split]),   
                "estimate.parquet"   
            ) 
        print("influence_estimate_path:", self.influence_estimate_path)
        print("dirname:", os.path.dirname(self.influence_estimate_path))
        print("exists:", os.path.exists(os.path.dirname(self.influence_estimate_path)))
        os.makedirs(os.path.dirname(self.influence_estimate_path), exist_ok=True)
           
        
    def init_model_and_tokenizer(self):
        
        base_model_name_or_path = None
        with open(os.path.join(self.model_path, "adapter_config.json")) as f:
            base_model_name_or_path = json.load(f)["base_model_name_or_path"]
            
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}" # https://huggingface.co/allenai/OLMo-2-1124-7B-SFT/blob/main/tokenizer_config.json 

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token # TODO

            
            
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
        # tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.model.resize_token_embeddings(len(self.tokenizer))


        self.model = PeftModel.from_pretrained(self.model, self.model_path, is_trainable=True)
        self.model.to(self.device)
       
    @abstractmethod
    def run(self):
        pass
    @abstractmethod
    def get_config_string(self):
        pass

      
            
    def load_gradients(self, dataset, dataset_name, dataset_split, max_workers=32):
        base_path = os.path.join(self.gradient_cache_dir, dataset_name, dataset_split)
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"No gradients found at {base_path}")

        grad_files = sorted([f for f in os.listdir(base_path) if f.endswith(".pt")])
        if len(grad_files) != len(dataset):
            raise FileNotFoundError()
        def load_grad(file_name):
            path = os.path.join(base_path, file_name)
            with open(path, "rb") as f:
                return torch.load(f, map_location="cpu")
           

        # with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #     list_of_dicts = list(executor.map(load_grad, grad_files))
        list_of_dicts = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
     
            futures = [executor.submit(load_grad, file) for file in grad_files]

            for future in tqdm(as_completed(futures), total=len(futures)):
                list_of_dicts.append(future.result())

        # merge
        gradients_dict = {}
        for d in list_of_dicts:
            gradients_dict.update(d)
        
        return gradients_dict

        
    def save(self):
        print("self.train_dataset_name",self.train_dataset_name)
        self.influence_estimate.columns = self.train_dataset["indices"]
        self.influence_estimate.to_parquet(self.influence_estimate_path)
        logger.info(f"Saved influence estimate to disk")
    def run_cached(self):
        self.influence_estimate = None
        try:
            self.influence_estimate =  pd.read_parquet(self.influence_estimate_path)
      
        except Exception as e:
            if not isinstance(e, FileNotFoundError):
                logger.error(f"Recomputing due to unexpected error while loading influence estimate: {e}", exc_info=True)
            self.init_model_and_tokenizer()
            self.run()
    def get_explanation(self, test_instance_idx):
        return self.influence_estimate.iloc[test_instance_idx]
    
    
    
    def get_gradient(self, dataset, dataset_name, dataset_split, train_instance_idx):

        grad_path = os.path.join(
            self.gradient_cache_dir,
            os.path.basename(dataset_name),
            dataset_split,
            f"gradient_{train_instance_idx}.pt"
        )
        
        if not os.path.exists(grad_path):
            raise FileNotFoundError(f"Gradient file not found: {grad_path}")
        return torch.load(grad_path)
def store_gradient(gradient_cache_dir, dataset_name, dataset_split, gradient_dict):
    base_path = os.path.join(gradient_cache_dir, dataset_name, dataset_split)
    os.makedirs(base_path, exist_ok=True)

    # need to handle bf16 conversion sperately for less and datainf formats
    gradient_dict_bf16 = {}
    for outer_k, v in gradient_dict.items():
        if isinstance(v, dict):
            # nested dict
            gradient_dict_bf16[outer_k] = {k: t.to(torch.bfloat16) for k, t in v.items()}
        elif isinstance(v, torch.Tensor):
            # flat dict
            gradient_dict_bf16[outer_k] = v.to(torch.bfloat16)
        else:
            raise ValueError(f"Unexpected type {type(v)} in gradient_dict")

    grad_path = os.path.join(base_path, f"gradient_{list(gradient_dict.keys())[0]}.pt")
    with open(grad_path, "wb") as f:
        torch.save(gradient_dict_bf16, f)

def gradient_exists(gradient_cache_dir, dataset_name, dataset_split, idx):
    base_path = os.path.join(gradient_cache_dir, dataset_name, dataset_split)
    grad_path = os.path.join(base_path, f"gradient_{idx}.pt")
    return os.path.exists(grad_path)
