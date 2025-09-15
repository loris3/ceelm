import os
import pickle
import logging
import torch
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
    def __init__(self, model_path, train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, device="cuda", proj_dim=2**13,fast_implementation=True):
        super().__init__(model_path, train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, device,
                         param_list=[proj_dim,fast_implementation],
                         )
        self.proj_dim = proj_dim
        self.fast_implementation = fast_implementation
        self.param_string = ""
        
        self.run_cached()    
    def get_config_string(self):
        return f"{self.__class__.__name__}: fast_implementation={str(self.fast_implementation)}"
    def run(self):
        self.lora_engine = LORAEngineGeneration(self.model, self.model_path, self.param_string, self.device, proj_dim=self.proj_dim)
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
        
        
        self.influence_estimate = self.influence_engine.IF_dict["proposed"]
        self.save()
        
    def get_gradients(self):

        
        try:
            self.tr_grad_dict = self.load_gradients(self.train_dataset, self.train_dataset_name, self.train_dataset_split)
        except (FileNotFoundError, RuntimeError):
            tokenized_train_dataset = tokenize_dataset(self.train_dataset, self.tokenizer)
          
            self.lora_engine.compute_gradient(tokenized_train_dataset, self.tokenizer, self.train_dataset_name, self.train_dataset_split, self.gradient_cache_dir)        
            self.get_gradients()
            return
        try:
            self.test_grad_dict = self.load_gradients(self.test_dataset, self.test_dataset_name, self.test_dataset_split)
        except (FileNotFoundError, RuntimeError):
            tokenized_test_dataset = tokenize_dataset(self.test_dataset, self.tokenizer)
            self.lora_engine.compute_gradient(tokenized_test_dataset,  self.tokenizer, self.test_dataset_name, self.test_dataset_split, self.gradient_cache_dir)
            self.get_gradients()
            return
   
            
    def get_gradient(self, dataset, dataset_name, dataset_split, train_instance_idx):
        grads_dict = super().get_gradient(dataset, dataset_name, dataset_split, train_instance_idx)

        return  torch.cat([g.flatten() for g in list(grads_dict.values())[0].values()])

