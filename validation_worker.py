import os
import torch
from tqdm import tqdm
CACHE_DIR = "./cache/validation/"


def process(batch, gpu_id):
    """
    Worker function to process a batch of test sets on a specific GPU.
    """
    os.makedirs(CACHE_DIR,exist_ok=True)
    try:
        # Restrict this process to a single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        import torch
        from validation_engine import ValidationEngine  # import inside worker

        engine = ValidationEngine("EleutherAI/pythia-31m", device="cuda:0")
        results = []
        for ts_batch, test_sets, test_indices, estimator, explanation_type in tqdm(batch, desc=f"GPU {gpu_id}", leave=False,position=gpu_id):
          
            deltas =  [engine.score(explanation, test_sets) for (setting, explanation) in ts_batch]
            results.append([(setting,
                             estimator.get_config_string().replace(" ",""), # estimator
                             explanation,
                             explanation_type, # type
                             test_indices, 
                             delta[0].item(),
                             ) for (setting, explanation), delta in zip(ts_batch, deltas)])
        return results

    except Exception as e:
        import traceback, logging
        logging.error(f"Error in process for GPU {gpu_id}: {e}")
        logging.error(traceback.format_exc())
        raise
