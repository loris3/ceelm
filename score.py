


import os

import torch
import logging
logger = logging.getLogger("ignite.handlers.early_stopping.EarlyStopping")
logger.setLevel(logging.WARNING)


import torch



import torch
import itertools
from tqdm import tqdm
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


import itertools


import torch
import os


import pandas as pd
from concurrent.futures import as_completed

from explanations import KRandom, Self
    
import logging
logging.getLogger().setLevel(logging.WARNING)



import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import multiprocessing
from tqdm import tqdm
import itertools
import pandas as pd
import traceback



logging.basicConfig(level=logging.ERROR, format='%(asctime)s [%(levelname)s] %(message)s')



    # return results_local
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanation_type", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("--m", type=int, required=False)
    parser.add_argument("--k", type=int, required=False)
    parser.add_argument("--lambda_", type=float, required=False)

    args = parser.parse_args()
    print(args)

    multiprocessing.set_start_method('spawn', force=True)   
    torch.manual_seed(42)
    from load_experiment_data import (
    train_dataset_name,
    test_dataset_name,
    train_dataset_split,
    test_dataset_split,
    load_data_and_estimators,
    explanation_types,
    explanation_m,
    linear_coders,
    explanation_k,
    explanation_lambda,
    explanation_seed
    )
    train_dataset, test_dataset, estimators = load_data_and_estimators()

    for estimator in estimators:
            print(f"Processing estimator {estimator.get_config_string()} for {os.path.basename(estimator.model_path)}", flush=True)
            print(args)
            if ("BM25" in estimator.get_config_string()) and (("Helpful" in args.explanation_type) or ("Harmful" in args.explanation_type) ):
                print("skipping BM25",args)
                
                continue
            

            partial_results_dir =  os.path.join("./cache/scoring/partial/",
                estimator.get_config_string(),
                os.path.basename(estimator.model_path),
                train_dataset_name,
                train_dataset_split,
                test_dataset_name,
                test_dataset_split,
                "partial")

            explanations = []

            if args.explanation_type == "KRandom":
                for idx in range(len(test_dataset)):
                    explanations.append(KRandom(idx, estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=args.k, seed=args.seed))
            elif args.explanation_type == "Self":
                for idx in range(len(test_dataset)):
                    explanations.append(Self(idx))
                    
            else:
                for base in explanation_types:
                   
                    if args.explanation_type == base.__name__:
            
                        for idx in range(len(test_dataset)):
                            if "Facility" in args.explanation_type:
                                explanations.append(base(idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=args.k, m=args.m, lambda_=args.lambda_))
                            else:
                                explanations.append(base(idx, estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=args.k))
                            

            assert len(explanations) > 0, "Provide a valid class name as arg"

            print(f"Total explanations for this estimator: {len(explanations)}", flush=True)

            with tqdm(total=len(explanations), desc="Explanations", position=0) as pbar:
                for explanation in explanations:

                    
                    
                    ii = explanation.document_idx
                    if isinstance(explanation, Self):
                        #                      from test dataset!
                        examples_to_train_on = test_dataset.select([explanation.document_idx])
                        dataset_for_grad = test_dataset
                        split_name = test_dataset_split
                    else:
                        examples_to_train_on = train_dataset.select(explanation.documents)
                        dataset_for_grad = train_dataset
                        split_name = train_dataset_split
            
                    device = "cuda:0"
    

    
                    test_grad = None
                    A = None

                

                    

                    for linear_coder in linear_coders:
                        # check if already done
                        o = linear_coder(A, test_grad, device=device, metadata_only=True)
                        results_path = os.path.join(
                            partial_results_dir,
                            o.description,
                            explanation.description,
                            str(ii) + ".parquet",
                            )
                    
                        os.makedirs(os.path.dirname(results_path), exist_ok=True)
                        
                        if os.path.isfile(results_path):
                            # print(f"Skipping {ii}: parquet file exists: {results_path}", flush=True)
                            continue
                        else:
       
                            if test_grad is None: # load once
                                test_grad = estimator.get_gradient(

                                    os.path.basename(test_dataset_name),
                                    test_dataset_split,
                                    ii,
                                ).to(device).to(torch.float32)
                                
                                
                            if A is None:    
                                if not isinstance(explanation, Self):
                                  
                                    A = torch.stack(
                                        estimator.get_gradient(
                 
                                            os.path.basename(train_dataset_name),
                                            train_dataset_split,
                                            explanation.documents,
                                        )
                                    ).to(torch.float32).to(device)
                                else:
                                    A = test_grad.unsqueeze(0)
                            o = linear_coder(A, test_grad, device=device, metadata_only=False, estimator_config=estimator.get_config_string())
                            
                            try:
                                o.fit()
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                    
                                
                                
                            x_hat_method = o.A.T @ o.t
                            
                            
                            
                            epsilon = 1e-8 
                            
                            var_pred_error_method = torch.var(test_grad - x_hat_method, correction=0)
                            pred_gain = torch.var(test_grad, correction=0) / (var_pred_error_method + epsilon)
                            
                            mse = torch.mean((test_grad - x_hat_method) ** 2)
                            
                            
                            l1 = torch.abs(o.t).sum() 
                            l2 = torch.square(o.t).sum()
                            

                                
                            df = pd.DataFrame([(
                                explanation.description,
                                os.path.basename(estimator.model_path),
                                estimator.get_config_string(),
                                explanation.document_idx,
                                train_dataset_name,
                                train_dataset_split,
                                test_dataset_name,
                                test_dataset_split,
                                o.description,
                                                            
                                pred_gain.item(), 
                                mse.item(),
                                l1.item(),
                                l2.item(),
                                
                            
                                o.t.cpu().tolist(),
                                
                                                
                                                
                                )], columns=[
                                "explanation_type", 
                                "model", 
                                "estimator",
                                "document_idx", 
                                "train_dataset", 
                                "train_split", 
                                "test_dataset", 
                                "test_split", 
                                "linear_coder",
                                
                                "pred_gain",
                                "mse",
                                
                                "l1",
                                "l2",
                                
                
                                "t"
                            ])
                            del o
                            assert df.notnull().all().all(), "DataFrame contains missing values"
                            assert not df.isnull().values.any(), "DataFrame contains NaN values"

                            df.to_parquet(results_path, index=False)
                    
                                    
                    
                    
                    
                    pbar.update(1)
                    pbar.refresh()


