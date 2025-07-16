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
    def __init__(self, model_path, adapter_path, train_dataset, test_dataset, tokenizer_path=None, device="cuda"):
        super().__init__(model_path, adapter_path, train_dataset, test_dataset, tokenizer_path, device)
        self.run_cached()    
    def run(self):
        self.lora_engine = LORAEngineGeneration(self.model, self.device)
        self.lora_engine.tokenizer = self.tokenizer
        self.influence_engine = IFEngineGeneration()
        
        self.get_gradients()
        logger.info(f"Preprocessing gradients...")
        self.influence_engine.preprocess_gradients(self.tr_grad_dict, self.test_grad_dict)
        logger.info(f"Computing HVPs...")
        self.influence_engine.compute_hvps()
        logger.info(f"Computing IF...")
        self.influence_engine.compute_IF()
        self.influence_estimate = self.influence_engine.IF_dict["proposed"]
        self.save()
        
    def get_gradients(self):
        tokenized_datasets = {
            "train": tokenize_dataset(self.train_dataset, self.tokenizer),
            "test": tokenize_dataset(self.test_dataset, self.tokenizer),
        }

        collate_fn = lambda x: self.tokenizer.pad(x, padding="longest", return_tensors="pt")
        grad_dicts = {}

        os.makedirs('./cache/datasets', exist_ok=True)

        for split in ["train", "test"]:
            fingerprint = tokenized_datasets[split]._fingerprint
            cache_path = os.path.join("./cache/gradients/datainf", f"{fingerprint}.pkl")

            if os.path.exists(cache_path):
                logger.info(f"Loading {split} gradients from {cache_path}")
                with open(cache_path, "rb") as f:
                    grad_dicts[split] = pickle.load(f)
            else:
                logger.info(f"No cache for {split}. Computing gradients...")
                grads = self.lora_engine.compute_gradient(tokenized_datasets[split], collate_fn)
                grad_dicts[split] = grads
                with open(cache_path, "wb") as f:
                    pickle.dump(grad_dicts[split], f)
                    logger.info(f"Saved {split} gradients to {cache_path}")

        self.tr_grad_dict = grad_dicts["train"]
        self.test_grad_dict = grad_dicts["test"]
       

