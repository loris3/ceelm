import torch
import itertools
from tqdm import tqdm
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
CACHE_DIR = "./cache/score/"

def process_single(estimator, explanation, train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, methods, device_id):
    device = f"cuda:{device_id}"

    test_grad = estimator.get_gradient(test_dataset, os.path.basename(test_dataset_name), test_dataset_split, explanation.dataset_idx).to(device)

    train_grads_cpu = torch.stack(
        [estimator.get_gradient(train_dataset, os.path.basename(train_dataset_name), train_dataset_split, i).cpu() for i in explanation.documents]
    )
    mean = train_grads_cpu.mean(dim=0, keepdim=True)
    centered = train_grads_cpu - mean
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    klt_basis = Vh.T.to(device)
    train_grads = train_grads_cpu.to(device)

    # KLT
    coeffs = klt_basis.T @ test_grad.view(-1)
    x_hat = klt_basis @ coeffs
    var_pred_error = torch.var(test_grad - x_hat, correction=0)
    pred_gain_klt = torch.var(test_grad, correction=0) / var_pred_error
    
    results_local = []
    
    os.makedirs(CACHE_DIR,exist_ok=True)
    for method in methods:
        idx = (explanation.__class__.__name__,
             os.path.basename(estimator.model_path),
             estimator.get_config_string().replace(" ",""),
             explanation.dataset_idx,
             train_dataset._fingerprint,
             test_dataset._fingerprint,
             method.__name__)
        file_path = os.path.join(CACHE_DIR, "_".join(str(x) for x in idx) + ".pkl")
        normalized_gain = None
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                normalized_gain = pickle.load(f)
        else:
                
            o = method(train_grads, test_grad, device=device)
            x_hat_method = o.train_grads.T @ o.factors
            var_pred_error_method = torch.var(test_grad - x_hat_method, correction=0)
            pred_gain = torch.var(test_grad, correction=0) / var_pred_error_method
            normalized_gain = pred_gain / pred_gain_klt
            del o
            with open(file_path, "wb") as f:
                    pickle.dump(normalized_gain, f)

        
            
        results_local.append(
            (*idx,
             normalized_gain.item())
        )
    return results_local


def process_estimator_group(estimator_group, train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, methods, device_id, max_workers=2):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        estimator, explanations = estimator_group
        for explanation in tqdm(explanations, total=len(explanations),desc="Explanations", position=3,leave=False):
            futures.append(
                    executor.submit(
                        process_single, estimator, explanation, 
                        train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, 
                        methods, device_id
                    )
                )
        for f in as_completed(futures):
            results.extend(f.result())
    return results
