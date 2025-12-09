import os
import torch
from tqdm import tqdm
import multiprocessing
import traceback
import argparse


from load_experiment_data import (
    train_dataset_name,
    test_dataset_name,
    train_dataset_split,
    test_dataset_split,
    load_data_and_estimators,
    explanation_types,
    explanation_m,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanation_type", type=str, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--lambda_", type=float, required=False)
    parser.add_argument("--m", type=int, required=True)
    args = parser.parse_args()
    print(args.explanation_type)

    multiprocessing.set_start_method('spawn', force=True)   
    torch.manual_seed(42)
    

    
    train_dataset, test_dataset, estimators = load_data_and_estimators()
    k = args.k
    lambda_ = args.lambda_
    m=args.m
    for estimator in estimators:
            print("estimator.__class__.__name__",estimator.__class__.__name__)
            if ("BM25" in estimator.get_config_string()) and (("Helpful" in args.explanation_type) or ("Harmful" in args.explanation_type) ):
                print("skipping BM25",args)
                continue
            print(f"Processing estimator: {os.path.basename(estimator.model_path)}", flush=True)


        
            for base in explanation_types:
                if args.explanation_type == base.__name__:
                    if k > m:
                        continue
                    for idx in tqdm( range(len(test_dataset)), desc=f"Test samples (k={k}, m={m})", leave=False):
                        try:
                            if "Facility" in args.explanation_type:
                                selection = base(idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=k, m=m, lambda_=lambda_)
                                selection.documents
                                del selection
                            if "DIVINE" in args.explanation_type:
                                selection = base(idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=k, m=m)
                                selection.documents
                                del selection
                            if "AIDE" in args.explanation_type:
                                selection = base(idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=k, m=m)
                                selection.documents
                                del selection
                                

                        except Exception as e:
                            print(f"Error at idx={idx}, k={k}, m={m}: {e}, lambda={lambda_}")
                            traceback.print_exc()
                            raise e
                            
    print("done!")

