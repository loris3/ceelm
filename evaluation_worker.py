import torch
import itertools
from tqdm import tqdm
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_explanation(partial_results_dir, estimator, explanation, train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, linear_coders, device_id, ii):
    device = f"cuda:{device_id}"
    

    
    test_grad = None
    A = None

   
    results_local = []
    

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
            print(f"Skipping {ii}: parquet file exists: {results_path}", flush=True)
            continue
        else:
            # run
            if test_grad is None: # load once
                test_grad = estimator.get_gradient(test_dataset, os.path.basename(test_dataset_name), test_dataset_split, explanation.document_idx).to(device).to(torch.float32)
            if A is None: # load once
                A = torch.stack(
                    [estimator.get_gradient(train_dataset, os.path.basename(train_dataset_name), train_dataset_split, i).to(torch.float32) for i in explanation.documents]
                ).to(device)
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
      

    return results_local
