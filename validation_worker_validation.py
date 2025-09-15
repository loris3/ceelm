import os
import torch
from tqdm import tqdm
import pandas as pd

def process(partial_results_dir, estimator, explanation, examples_to_train_on, indices_to_train_on, examples_to_test_on, indices_to_test_on, train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, device_id, seed, ii):
        results_path = os.path.join(
            partial_results_dir,
            str(seed),
            str(ii) + ".parquet",
            )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if os.path.isfile(results_path):
            print(f"Skipping {ii}: parquet file exists", flush=True)
            return
        else:
            try:
                # Restrict this process to a single GPU
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
                import torch
                from validation_engine import ValidationEngine  # import inside worker

                engine = ValidationEngine(
                                          estimator.model_path, 
                                          device="cuda")
                
                delta = engine.score(examples_to_train_on, examples_to_test_on, seed=seed+ii)
                
                delta_target_document = delta[0].item()
               # delta_random_documents = delta[1:].tolist()
                
                
                df = pd.DataFrame([(
                        explanation.description,
                        os.path.basename(estimator.model_path),
                        estimator.get_config_string(),
                        explanation.document_idx,
                        train_dataset_name,
                        train_dataset_split,
                        test_dataset_name,
                        test_dataset_split,
                        
                        indices_to_train_on,
                        
                        indices_to_test_on[0],
                #        indices_to_test_on[1:],
                        
                       # delta_random_documents,
                        delta_target_document
                    )], columns=[
                        "explanation_type", 
                        "model", 
                        "estimator",
                        "document_idx", 
                        "train_dataset", 
                        "train_split", 
                        "test_dataset", 
                        "test_split", 
                        
                        "indices_trained_on",
                        "indices_target_document",
                  #      "indices_random_documents",
                    
                        
                 #       "delta_random_documents",
                        "delta_target_document"
                        
                ])
                            
                assert df.notnull().all().all(), "DataFrame contains missing values"
                assert not df.isnull().values.any(), "DataFrame contains NaN values"
                df.to_parquet(results_path, index=False)
        
        
        

            except Exception as e:
                import traceback, logging
                logging.error(f"Error in process for GPU {device_id}: {e}")
                logging.error(traceback.format_exc())
                raise
        return []
