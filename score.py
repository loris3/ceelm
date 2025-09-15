import os

import torch
import logging
logger = logging.getLogger("ignite.handlers.early_stopping.EarlyStopping")
logger.setLevel(logging.WARNING)


import torch




import itertools


import torch
import os


import pandas as pd
from concurrent.futures import as_completed


    
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




if __name__ == "__main__":

    multiprocessing.set_start_method('spawn', force=True)   
    torch.manual_seed(42)
    from load_experiment_data import (
    train_dataset_name,
    test_dataset_name,
    train_dataset_split,
    test_dataset_split,
    load_data_and_estimators,
    explanation_types,
    linear_coders,
    
    )
    train_dataset, test_dataset, estimators = load_data_and_estimators()

    for estimator in estimators:
        print(estimator.get_config_string(), estimator.get_gradient(train_dataset, train_dataset_name, train_dataset_split, 0).shape)




    num_devices = torch.cuda.device_count()




    import evaluation_worker
    device_ids = itertools.cycle(range(num_devices))
    results = []
    import logging



    logging.getLogger("wandb").setLevel(logging.ERROR)
    import os
    # os.environ["WANDB_SILENT"] = "true"
    
    
    import wandb
    wandb.init(
        project="linear_coder_scheduler"
    )




    explanation_queue = []


    for estimator in estimators:
        partial_results_dir =  os.path.join("./cache/scoring/partial/",
            estimator.get_config_string(),
            os.path.basename(estimator.model_path),
            train_dataset_name,
            train_dataset_split,
            test_dataset_name,
            test_dataset_split,
            "partial")

        explanations = [
            explanation_type(idx, estimator)
            for explanation_type in explanation_types
            for idx in range(len(test_dataset))
        ]

        explanation_queue.extend([
            (explanation, partial_results_dir) for explanation in explanations
        ])
        
        
        
    with ProcessPoolExecutor(max_workers=num_devices) as executor:
        futures = {
            executor.submit(
                evaluation_worker.process_explanation,
                partial_results_dir,
                explanation.estimator, explanation, 
                train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, 
                linear_coders,
                next(device_ids),
                explanation.document_idx
            ): explanation for explanation, partial_results_dir in explanation_queue
        }

        with tqdm(total=len(futures), desc="Explanations", position=0) as pbar:
            for future in as_completed(futures):
                try:
                    future.result()  
                    wandb.log({"_": None})
                except Exception as e:
                    logging.error(f"A future failed: {e}\n{traceback.format_exc()}")
                    raise
                finally:
                    pbar.update(1)
                
        wandb.finish()