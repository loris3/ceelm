import os
import pickle
import logging

from influence_estimation.estimator import BaseEstimator
from influence_estimation.datainf.lora_model import LORAEngineGeneration
from influence_estimation.datainf.influence import IFEngineGeneration
from finetune import tokenize_dataset
import util

# Configure logger at module level
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class DataInfEstimator(BaseEstimator):
    def __init__(self, model_path, adapter_path, train_dataset, test_dataset, tokenizer_path=None, device="cuda", proj_dim=8192,fast_implementation=True):
        super().__init__(model_path, adapter_path, train_dataset, test_dataset, tokenizer_path, device,
                         param_list=[proj_dim,fast_implementation],
                         )
        self.proj_dim = proj_dim
        self.fast_implementation = fast_implementation
        self.run_cached()    
    def get_config_string(self):
        return f"{self.__class__.__name__}: fast_implementation={str(self.fast_implementation)}"
    def run(self):
        self.lora_engine = LORAEngineGeneration(self.model, self.device, self.proj_dim)
        self.lora_engine.tokenizer = self.tokenizer
        self.influence_engine = IFEngineGeneration()
        
        self.get_gradients()
        logger.info(f"Preprocessing gradients...")
        self.influence_engine.preprocess_gradients(self.tr_grad_dict, self.test_grad_dict)
        
        
        if self.fast_implementation:
            logger.info(f"Computing HVPs...")
            self.influence_engine.compute_hvps()
            logger.info(f"Computing IF...")
            self.influence_engine.compute_IF()
        else:
            logger.info(f"Computing HVPs (slow)...")
            self.influence_engine.compute_hvps_slow()
            logger.info(f"Computing IF (slow)...")
            self.influence_engine.compute_IF_slow()
        
        
        self.influence_estimate = -self.influence_engine.IF_dict["proposed"]
        self.save()
        
    def get_gradients(self):
        collate_fn = lambda x: self.tokenizer.pad(x, padding="longest", return_tensors="pt")

        tokenized_train_dataset = tokenize_dataset(self.train_dataset, self.tokenizer)
        try:
            self.tr_grad_dict = self.load_gradients(tokenized_train_dataset)
        except (FileNotFoundError, RuntimeError):
            if self.fast_implementation:
                self.tr_grad_dict = self.lora_engine.compute_gradient(tokenized_train_dataset, collate_fn)
            else:
                self.tr_grad_dict = self.lora_engine.compute_gradient_slow(tokenized_train_dataset, collate_fn)
            self.store_gradients(tokenized_train_dataset, self.tr_grad_dict)
            
        tokenized_test_dataset = tokenize_dataset(self.test_dataset, self.tokenizer)
        try:
            self.test_grad_dict = self.load_gradients(tokenized_test_dataset)
        except (FileNotFoundError, RuntimeError):
            if self.fast_implementation:
                self.test_grad_dict = self.lora_engine.compute_gradient(tokenized_test_dataset, collate_fn)
            else: 
                self.test_grad_dict = self.lora_engine.compute_gradient_slow(tokenized_test_dataset, collate_fn)
            self.store_gradients(tokenized_test_dataset, self.test_grad_dict)
            
    