
from itertools import chain
from influence_estimation.data_inf import DataInfEstimator
from influence_estimation.less_inf import LESSEstimator
from datasets import load_dataset

import torch
from scipy import stats
import numpy as np

import numpy as np
from scipy import stats
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
MODEL = "OLMo-2-0425-1B_tulu-v2-sft-mixture"
indices_dir = os.path.join("./models", MODEL, "indices")
indices = [
    (*f.split("_")[:-1], torch.load(os.path.join(indices_dir, f)))
    for f in os.listdir(indices_dir)
    if os.path.isfile(os.path.join(indices_dir, f)) and f.endswith((".pt"))
]
indices = [(int(i.replace("iter","")),t, [int(j) for j in ind]) for i,t, ind in indices]
df = pd.DataFrame(indices, columns=["interation", "subset", "indices"])
df
# base_model_path = "distilbert/distilgpt2"
base_model_path = "allenai/OLMo-2-0425-1B"
# adapter_path = "/mnt/nlp-data/home/users/loriss21cs/cfe/models/distilgpt2_tulu-v2-sft-mixture"
adapter_path = "/mnt/nlp-data/home/users/loriss21cs/cfe/models/OLMo-2-0425-1B_tulu-v2-sft-mixture"

train_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train")#.shuffle(seed=0).select(range(20))





test_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train").shuffle(seed=0).select(range(100))


class Experiment:
    def __init__(self, full_train_dataset, model_path="./models/OLMo-2-0425-1B_tulu-v2-sft-mixture"):
        self.full_train_dataset = full_train_dataset
        def add_index(example, idx):
            example["indices"] = idx
            return example
        self.full_train_dataset = self.full_train_dataset.map(add_index, with_indices=True, num_proc=10)
            
        self.model_path = model_path
        self.train_indices_path = os.path.join(model_path, "indices")
        self.load_indices()
        self._coreset_dataset = None  # Private backing field
        self._train_dataset = None  # Private backing field
    def load_indices(self):
        indices = [
        (*f.split("_")[:-1], torch.load(os.path.join(self.train_indices_path, f)))
            for f in os.listdir(indices_dir)
                if os.path.isfile(os.path.join(indices_dir, f)) and f.endswith((".pt"))
        ]
        indices = [(int(i.replace("iter","")),t, [int(j) for j in ind]) for i,t, ind in indices]
        df = pd.DataFrame(indices, columns=["iteration", "subset", "indices"]).sort_values(by="iteration")        
        self.train_indices = {
            subset: {
                iteration: group[group["subset"] == subset]["indices"].tolist()[0]
                for iteration, group in df.groupby("iteration")
            }
            for subset in df["subset"].unique()  
        }
    @property
    def coreset_dataset(self):
        if self._coreset_dataset is None:           
            self._coreset_dataset = self.full_train_dataset.select(
                set(chain.from_iterable(self.train_indices["selected"].values()))
            )
        return self._coreset_dataset
    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = self.full_train_dataset.select(
                set(chain.from_iterable(self.train_indices["full"].values()))
            )
        return self._train_dataset
    
    
    

if __name__ == "__main__":
    from multiprocess import set_start_method
    set_start_method("spawn")
    experiment = Experiment(train_dataset)     
            
    estimators = [
        # DataInfEstimator(base_model_path, adapter_path, train_dataset, test_dataset,fast_implementation=False),
        # DataInfEstimator(base_model_path, adapter_path, train_dataset, test_dataset,fast_implementation=True),
        LESSEstimator(base_model_path, adapter_path, experiment.train_dataset, test_dataset,normalize=True),
        LESSEstimator(base_model_path, adapter_path, experiment.coreset_dataset, test_dataset,normalize=True),
        # LESSEstimator(base_model_path, adapter_path, train_dataset, test_dataset,normalize=False)
    ]
    for estimator in estimators:
        estimator.init_model_and_tokenizer()
        estimator.get_gradients(estimator.train_dataset, gradient_type="adam")