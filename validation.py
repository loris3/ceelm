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
    linear_coders,
    explanation_k,
    
)
from explanations import KRandom, Self

logging.basicConfig(level=logging.ERROR, format='%(asctime)s [%(levelname)s] %(message)s')


def process(
    engine,
    partial_results_dir,
    estimator,
    explanation,
    examples_to_train_on,
    indices_to_train_on,
    examples_to_test_on,
    indices_to_test_on,
    train_dataset,
    train_dataset_name,
    train_dataset_split,
    test_dataset,
    test_dataset_name,
    test_dataset_split,
    ii
):
    results_path = os.path.join(
        partial_results_dir,
        explanation.description,
        f"{ii}.parquet",
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    if os.path.isfile(results_path):
        print(f"Skipping {ii}: parquet file exists", flush=True)
        return

    try:
        # score delta
        delta = engine.score(examples_to_train_on, examples_to_test_on, seed=42)
        delta_target_document = delta[0].item()

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
            "delta_target_document"
        ])
        # if everything before max_len is masked (i.e. "user"), log_p will be nan -> delta nan
        # we still write the result and filter these instances later

        df.to_parquet(results_path, index=False)

    except Exception as e:
        import traceback
        logging.error(f"Error processing document {ii}: {e}")
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanation_type", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False)
    args = parser.parse_args()

    torch.manual_seed(42)

    train_dataset, test_dataset, estimators = load_data_and_estimators()

    print("Total estimators:", len(estimators), flush=True)

    for estimator in estimators:
        print(f"Processing estimator: {os.path.basename(estimator.model_path)}", flush=True)

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
            for k in explanation_k:
                for idx in range(len(test_dataset)):
                    explanations.append(KRandom(idx, estimator, k=k, seed=args.seed))
        elif args.explanation_type == "Self":
            for idx in range(len(test_dataset)):
                explanations.append(Self(idx, estimator, k=1))
        else:
            for base in explanation_types:
                if args.explanation_type == base.__name__:
                    for k in explanation_k:
                        for idx in range(len(test_dataset)):
                            explanations.append(base(idx, estimator, k=k))

        assert len(explanations) > 0, "Provide a valid class name as arg"

        print(f"Total explanations for this estimator: {len(explanations)}", flush=True)

        with tqdm(total=len(explanations), desc="Explanations", position=0) as pbar:
            for explanation in explanations:
                process(
                    engine,
                    partial_results_dir,
                    estimator,
                    explanation,
                    train_dataset.select(explanation.documents),
                    explanation.documents,
                    test_dataset.select([explanation.document_idx]),
                    [explanation.document_idx],
                    train_dataset, train_dataset_name, train_dataset_split,
                    test_dataset, test_dataset_name, test_dataset_split,
                    explanation.document_idx
                )
                pbar.update(1)
                pbar.refresh()
