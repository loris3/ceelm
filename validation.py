import os
import torch
from tqdm import tqdm
import pandas as pd
import logging
import argparse
from validation_engine import ValidationEngine

from load_experiment_data import (
    train_dataset_name,
    test_dataset_name,
    train_dataset_split,
    test_dataset_split,
    load_data_and_estimators,
    explanation_types,   
)
from explanations import KRandom, Self

logging.basicConfig(level=logging.ERROR, format='%(asctime)s [%(levelname)s] %(message)s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanation_type", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("--m", type=int, required=False)
    parser.add_argument("--k", type=int, required=False)
    parser.add_argument("--lambda_", type=float, required=False)
    args = parser.parse_args()

    torch.manual_seed(42)

    train_dataset, test_dataset, estimators = load_data_and_estimators()

    print("Total estimators:", len(estimators), flush=True)

    for estimator in estimators:
        print(f"Processing estimator {estimator.get_config_string()} for {os.path.basename(estimator.model_path)}", flush=True)
        print(args)
        if ("BM25" in estimator.get_config_string()) and (("Helpful" in args.explanation_type) or ("Harmful" in args.explanation_type) ):
            print("skipping BM25",args)
            continue

        # Create engine once per estimator
        engine = ValidationEngine(estimator.model_path)

        partial_results_dir = os.path.join(
            "./cache/validation/partial/",
            estimator.get_config_string(),
            os.path.basename(estimator.model_path),
            train_dataset_name,
            train_dataset_split,
            test_dataset_name,
            test_dataset_split,
            
        )

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
                            elif "DIVINE" in args.explanation_type:
                                explanations.append(base(idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=args.k, m=args.m))
                            elif "AIDE" in args.explanation_type:
                                explanations.append(base(idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=args.k, m=args.m))
                            else:
                                explanations.append(base(idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=args.k))

        assert len(explanations) > 0, "Provide a valid class name as arg"

        print(f"Total explanations for this estimator: {len(explanations)}", flush=True)

        with tqdm(total=len(explanations), desc="Explanations", position=0) as pbar:
            for explanation in explanations:
                examples_to_train_on = None
                examples_to_test_on = None
                if isinstance(explanation, Self):
                    #                      from test dataset!
                    examples_to_train_on = test_dataset.select([explanation.document_idx])
                    indices_to_train_on = [explanation.document_idx]
                    
                    examples_to_test_on = examples_to_train_on
                    indices_to_test_on = indices_to_train_on
                else:
                    examples_to_train_on = train_dataset.select(explanation.documents)
                    indices_to_train_on = explanation.documents
                    
                    examples_to_test_on = test_dataset.select([explanation.document_idx])
                    indices_to_test_on = [explanation.document_idx]

                results_path = os.path.join(
                partial_results_dir,
                explanation.description,
                f"{explanation.document_idx}.parquet",
                )
                os.makedirs(os.path.dirname(results_path), exist_ok=True)

                if os.path.isfile(results_path):
                    print(f"Skipping {explanation.document_idx}: parquet file exists", flush=True)
                    continue

                try:
                    # score delta
                    metrics = engine.score(examples_to_train_on, examples_to_test_on, seed=42)
                    print(metrics["delta_log_p"])

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
                        metrics["delta_log_p"][0].item(),
                        metrics["log_p_before_ft"][0].item(),
                        metrics["log_p_after_ft"][0].item(),
                        metrics["jsd"][0].item(),
                        metrics["kld(before||after)"][0].item()
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
                        "delta_log_p",
                        "log_p_before",
                        "log_p_after",
                        "jsd",
                        "kld(before||after)"
                    ])
                    # if everything before max_len is masked (i.e. "user"), log_p will be nan -> delta nan
                    # we still write the result and filter these instances later

                    df.to_parquet(results_path, index=False)

                except Exception as e:
                    import traceback
                    logging.error(f"Error processing document {explanation.document_idx}: {e}")
                    logging.error(traceback.format_exc())
                    raise
                pbar.update(1)
                pbar.refresh()
