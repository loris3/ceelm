from abc import ABC, abstractmethod
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
import os
import logging
import pandas as pd
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

 

class BaseEstimator(ABC):
    def __init__(self, model_path, adapter_path, train_dataset, test_dataset, tokenizer_path=None, device="cuda"):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else self.adapter_path
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset     
        self.device = device
        
        os.makedirs(os.path.join('./results/influence',str(self.__class__.__name__)), exist_ok=True)
        self.influence_estimate_path = os.path.join(
                "./results/influence",
                self.__class__.__name__,
                "_".join(
                [
                    os.path.basename(self.model_path),
                    self.train_dataset._fingerprint,
                    self.test_dataset._fingerprint,
                ]
                ) + ".parquet"        
            ) 
           
            
    def init_model_and_tokenizer(self):
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path, is_trainable=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self.model.to(self.device)    
    @abstractmethod
    def run(self):
        pass
    def save(self):
        self.influence_estimate.to_parquet(self.influence_estimate_path)
        logger.warning(f"Saved influence estimate to disk")
    def run_cached(self):
        self.influence_estimate = None
        try:
            self.influence_estimate =  pd.read_parquet(self.influence_estimate_path)
            logger.warning(f"Loaded influence estimate from disk")
        except Exception as e:
            if not isinstance(e, FileNotFoundError):
                logger.error(f"Recomputing due to unexpected error while loading influence estimate: {e}", exc_info=True)
            self.init_model_and_tokenizer()
            self.run()