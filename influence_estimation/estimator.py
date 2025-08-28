from abc import ABC, abstractmethod
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
import os
import logging
import torch
import pandas as pd
import json 


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class BaseEstimator(ABC):
    def __init__(self, model_path, train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, device="cuda", param_list=[]):
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
        
        self.gradient_cache_dir = os.path.join("./cache/gradients/full", self.__class__.__name__, self.param_string, os.path.basename(self.model_path))
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
    def store_gradients(self, dataset, dataset_name, dataset_split, gradients):
        outp_path = os.path.join(self.gradient_cache_dir, dataset_name, dataset_split, "gradients.pt")
        os.makedirs(os.path.dirname(outp_path), exist_ok=True)
        torch.save(gradients, outp_path)
        
    def load_gradients(self, dataset, dataset_name, dataset_split):
        outp_path = os.path.join(self.gradient_cache_dir, dataset_name, dataset_split, "gradients.pt")
        return torch.load(outp_path)
        
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
        if (dataset_name, dataset_split) not in self.gradient_cache:
            self.gradient_cache[(dataset_name, dataset_split)] = self.load_gradients(dataset, dataset_name, dataset_split)
        return self.gradient_cache[(dataset_name, dataset_split)][train_instance_idx]
