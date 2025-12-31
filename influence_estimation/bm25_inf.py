import os
import math
import logging
from collections import Counter, defaultdict
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json 
from transformers import AutoTokenizer
from influence_estimation.estimator import BaseEstimator
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
logger = logging.getLogger(__name__)
from functools import partial
import torch
N_PROC=8



class BM25Estimator(BaseEstimator):
    def __init__(self, model_path, train_dataset, train_dataset_name, train_dataset_split,
                 test_dataset, test_dataset_name, test_dataset_split, device="cpu",
                 k1=1.5, b=0.75, eval_mode=False, num_workers=None):
        super().__init__(model_path, train_dataset, train_dataset_name, train_dataset_split,
                         test_dataset, test_dataset_name, test_dataset_split,
                         device=device, param_list=[k1, b], eval_mode=eval_mode)
        self.gradient_cache_dir = os.path.join("/tmp/cache/gradients/full", "   ","8192-True", os.path.basename(self.model_path))
        self.gradient_out_dir = os.path.join("cache/gradients/full", "DataInfEstimator","8192-True", os.path.basename(self.model_path))
        if eval_mode:
            self.gradient_cache_dir = self.gradient_out_dir
        self.k1 = k1
        self.b = b
        self.num_workers = num_workers or N_PROC
        self.run_cached()

    def get_config_string(self):
        return f"{self.__class__.__name__}: k1={self.k1}, b={self.b}"

    def run_cached(self):
        self.influence_estimate = None
        try:
            self.influence_estimate = pd.read_parquet(self.influence_estimate_path)
            logger.info(f"Loaded cached BM25 estimate from {self.influence_estimate_path}")
        except Exception:
            base_model_name_or_path = None
            with open(os.path.join(self.model_path, "adapter_config.json")) as f:
                base_model_name_or_path = json.load(f)["base_model_name_or_path"]
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
            chat_template_path="./chat_template.jinja"
            self.tokenizer.chat_template = open(chat_template_path).read()
            self.run()

    def run(self):
    
        def _tokenize_fn(batch, idx=None):
            # list of dicts: [{"role": "user", "content": ...}, {"role": "assistant", ...}]
            texts = [
                self.tokenizer.apply_chat_template(conv, tokenize=False)
                for conv in batch["messages"]
            ]
            enc = self.tokenizer(
                texts,
                truncation=True,
                max_length=4096,
                return_attention_mask=True,
            )         
               
            return enc

        train_ds = self.train_dataset.map(
            _tokenize_fn,
            with_indices=True,
            batched=True,
            num_proc=N_PROC,
            remove_columns=[col for col in self.train_dataset.column_names if col != "messages"],
        )
    
       

        train_tokens = train_ds["input_ids"]
        train_indices = train_ds["indices"] if "indices" in train_ds.column_names else list(range(len(train_ds)))

        test_ds = self.test_dataset.map(
            _tokenize_fn,
            with_indices=True,
            batched=True,
            num_proc=N_PROC,
            remove_columns=[col for col in self.test_dataset.column_names if col != "messages"],
        )
        test_tokens = test_ds["input_ids"]
        test_indices = test_ds["indices"] if "indices" in test_ds.column_names else list(range(len(test_ds)))


        n_train = len(train_tokens)
        doc_lengths = [len(doc) for doc in train_tokens]
        avgdl = sum(doc_lengths) / max(1, n_train)

        df = defaultdict(int)
        tf_per_doc = []
        for doc in train_tokens:
            c = Counter(doc)
            tf_per_doc.append(c)
            for term in c.keys():
                df[term] += 1

        N = max(1, n_train)
        idf = {term: math.log((N - freq + 0.5) / (freq + 0.5) + 1.0) for term, freq in df.items()}

        inv_index = {}
        for doc_idx, c in enumerate(tf_per_doc):
            for term, tfv in c.items():
                inv_index.setdefault(term, []).append((doc_idx, tfv))

        denom_cache = [1.0 + self.k1 * (1.0 - self.b + self.b * (dl / avgdl)) for dl in doc_lengths]


        def compute_scores(batch_tokens):
            batch_results = []
            for tokens in tqdm(batch_tokens, desc="Scoring tokens", leave=False):
                scores = [0.0] * n_train
                qf = Counter(tokens)
                for term, qfreq in qf.items():
                    if term not in inv_index:
                        continue
                    term_idf = idf.get(term, 0.0)
                    for doc_idx, tfv in inv_index[term]:
                        numer = tfv * (self.k1 + 1.0)
                        denom = tfv + denom_cache[doc_idx]
                        score = term_idf * numer / denom
                        scores[doc_idx] += score
                batch_results.append(scores)
            return batch_results


        results = np.zeros((len(test_tokens), len(train_tokens)), dtype=np.float32)

        # Decide on number of batches = number of workers
        num_batches = self.num_workers
        batch_size = (len(test_tokens) + num_batches - 1) // num_batches  # ceil division

        # Create larger batches
        batches = [(i, test_tokens[i:i + batch_size]) for i in range(0, len(test_tokens), batch_size)]

        def worker(batch_tokens, start_idx):
            batch_results = compute_scores(batch_tokens)
            return start_idx, batch_results

        # Submit only one future per batch
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(worker, batch_tokens, start_idx) for start_idx, batch_tokens in batches]
            
            for future in tqdm(futures, desc="BM25 scoring"):
                start_idx, batch_scores = future.result()
                for i, scores in enumerate(batch_scores):
                    results[start_idx + i, :] = scores


        df_scores = pd.DataFrame(results, index=test_indices, columns=train_indices)
        self.influence_estimate = df_scores
        self.save()
        logger.info("BM25 influence estimate computed and saved.")
    def get_gradient(self,  dataset_name, dataset_split, train_instance_idx):
        if isinstance(train_instance_idx, int):
            grads_dict = super().get_gradient(dataset_name, dataset_split, train_instance_idx)

            return  torch.cat([g.flatten() for g in list(grads_dict.values())[0].values()])
        else:
            def fetch_grad(idx, get_grad_fn, dataset_name, dataset_split):            
                grads_dict = get_grad_fn(dataset_name, dataset_split, idx)
                return  torch.cat([g.to_dense().flatten() for g in list(grads_dict.values())[0].values()])

          
            fetch_grad_partial = partial(fetch_grad, get_grad_fn=super().get_gradient, dataset_name=dataset_name, dataset_split=dataset_split)

            with ThreadPoolExecutor() as executor:
                return list(executor.map(fetch_grad_partial, train_instance_idx))

    def get_gradient_dict(self, dataset, dataset_name, dataset_split, train_instance_idx):
        if isinstance(train_instance_idx, int):
            grads_dict = super().get_gradient(dataset_name, dataset_split, train_instance_idx)
            return next(iter(grads_dict.values()))

        else:
            
            def fetch_grad(idx, get_grad_fn, dataset, dataset_name, dataset_split):
                        grads_dict = get_grad_fn(dataset, dataset_name, dataset_split, idx)
                        return next(iter(grads_dict.values()))

          
            fetch_grad_partial = partial(fetch_grad, get_grad_fn=super().get_gradient,
                                        dataset=dataset, dataset_name=dataset_name, dataset_split=dataset_split)

            with ThreadPoolExecutor() as executor:
                return list(executor.map(fetch_grad_partial, train_instance_idx))

     