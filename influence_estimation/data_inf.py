import os
import pickle
import logging
import torch
from influence_estimation.estimator import BaseEstimator
from influence_estimation.datainf.lora_model import LORAEngineGeneration
from influence_estimation.datainf.influence import IFEngineGeneration
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from finetune import tokenize_dataset
import util

# Configure logger at module level
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class DataInfEstimator(BaseEstimator):
    def __init__(self, model_path, train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, device="cuda", proj_dim=2**13,fast_implementation=True,eval_mode=False):
        super().__init__(model_path, train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, device,
                         param_list=[proj_dim,fast_implementation],eval_mode=eval_mode
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
        self.influence_engine.preprocess_gradients(self.test_grad_dict, self.get_gradient_dict)
        
        
        if self.fast_implementation:
            logger.info(f"Computing HVPs...")
            self.influence_engine.compute_hvps(self.tokenized_train_dataset, self.train_dataset_name, self.train_dataset_split)
            logger.info(f"Computing IF...")
            self.influence_engine.compute_IF(self.tokenized_train_dataset, self.train_dataset_name, self.train_dataset_split)
        else:
            logger.info(f"Computing HVPs (slow)...")
            self.influence_engine.compute_hvps_slow()
            logger.info(f"Computing IF (slow)...")
            self.influence_engine.compute_IF_slow(self.tokenized_train_dataset, self.train_dataset_name, self.train_dataset_split)
        
        
        self.influence_estimate = self.influence_engine.IF_dict["proposed"]
        self.save()
        
    def get_gradients(self):

        self.tokenized_train_dataset = tokenize_dataset(self.train_dataset, self.tokenizer)
        
        if not self.lora_engine.all_gradients_exist(self.train_dataset, self.train_dataset_name, self.train_dataset_split, self.gradient_cache_dir):
            self.lora_engine.compute_gradient(self.tokenized_train_dataset, self.tokenizer, self.train_dataset_name, self.train_dataset_split, self.gradient_cache_dir, self.gradient_out_dir)        
        else:
            print("train grads cached")

        try:
            self.test_grad_dict = self.load_gradients(self.test_dataset, self.test_dataset_name, self.test_dataset_split)
        except (FileNotFoundError, RuntimeError):
            tokenized_test_dataset = tokenize_dataset(self.test_dataset, self.tokenizer)
            self.lora_engine.compute_gradient(tokenized_test_dataset,  self.tokenizer, self.test_dataset_name, self.test_dataset_split, self.gradient_cache_dir, self.gradient_out_dir)
            self.get_gradients()
            return



    def get_gradient(self, dataset, dataset_name, dataset_split, train_instance_idx):
        grads_dict = super().get_gradient(dataset, dataset_name, dataset_split, train_instance_idx)

        return  torch.cat([g.flatten() for g in list(grads_dict.values())[0].values()])

    def get_gradient_dict(self, dataset, dataset_name, dataset_split, train_instance_idx):
        if isinstance(train_instance_idx, int):
            grads_dict = super().get_gradient(dataset, dataset_name, dataset_split, train_instance_idx)
            return next(iter(grads_dict.values()))

        elif isinstance(train_instance_idx, (list, tuple)):
            
            def fetch_grad(idx, get_grad_fn, dataset, dataset_name, dataset_split):
                        grads_dict = get_grad_fn(dataset, dataset_name, dataset_split, idx)
                        return next(iter(grads_dict.values()))

          
            fetch_grad_partial = partial(fetch_grad, get_grad_fn=super().get_gradient,
                                        dataset=dataset, dataset_name=dataset_name, dataset_split=dataset_split)

            with ThreadPoolExecutor() as executor:
                return list(executor.map(fetch_grad_partial, train_instance_idx))

        else:
            raise TypeError("train_instance_idx must be an int or a list/tuple of ints")