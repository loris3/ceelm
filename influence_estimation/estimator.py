from abc import ABC, abstractmethod
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
import os
import logging
import torch
import pandas as pd


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class BaseEstimator(ABC):
    def __init__(self, model_path, adapter_path, train_dataset, test_dataset, tokenizer_path=None, device="cuda", param_list=[]):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else self.adapter_path
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset     
        self.device = device
        self.param_string = "-".join([str(s) for s in param_list])
        
        
        self.gradient_cache_dir = os.path.join("./cache/gradients", self.__class__.__name__, self.param_string)
        os.makedirs(os.path.join('./results/influence',str(self.__class__.__name__), self.param_string), exist_ok=True)
        self.influence_estimate_path = os.path.join(
                "./results/influence",
                self.__class__.__name__,
                self.param_string,
                "_".join(
                [
                    os.path.basename(self.model_path),
                    self.train_dataset._fingerprint,
                    self.test_dataset._fingerprint,
                ]
                ) + ".parquet"        
            ) 
           
        
    def init_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model.resize_token_embeddings(len(self.tokenizer))  # as we do the same in training script (and we do not want to ship the full model)

        self.model = PeftModel.from_pretrained(self.model, self.adapter_path, is_trainable=True)
        self.model.to(self.device)
       
    @abstractmethod
    def run(self):
        pass
    @abstractmethod
    def get_config_string(self):
        pass
    def store_gradients(self, dataset, gradients):
        outp_path = os.path.join(self.gradient_cache_dir, dataset._fingerprint)
        os.makedirs(self.gradient_cache_dir, exist_ok=True)
        torch.save(gradients, outp_path)
        
    def load_gradients(self, dataset):
        outp_path = os.path.join(self.gradient_cache_dir, dataset._fingerprint)
        return torch.load(outp_path)
        
    def save(self):
        self.influence_estimate.columns = self.train_dataset["indices"]
        self.influence_estimate.to_parquet(self.influence_estimate_path)
        logger.warning(f"Saved influence estimate to disk")
    def run_cached(self):
        self.influence_estimate = None
        try:
            self.influence_estimate =  pd.read_parquet(self.influence_estimate_path)
      
        except Exception as e:
            if not isinstance(e, FileNotFoundError):
                logger.error(f"Recomputing due to unexpected error while loading influence estimate: {e}", exc_info=True)
            self.init_model_and_tokenizer()
            self.run()
            