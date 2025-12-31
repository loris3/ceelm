import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import os


import argparse
from influence_estimation.data_inf import DataInfEstimator
from influence_estimation.less_inf import LESSEstimator
from influence_estimation.bm25_inf import BM25Estimator

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run influence estimation scoring.")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="base model path"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="model path"
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        required=True,
       help="train dataset path"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        help="test dataset path"
    )
    
    parser.add_argument(
        "--train_dataset_split",
        type=str,
        required=True,
       help="e.g., train[0:100]"
    )
    parser.add_argument(
        "--test_dataset_split",
        type=str,
        required=True,
        help="e.g., train[0%:10%]"
    )
    return parser.parse_args()


def main():
    args = parse_args()


    train_dataset = load_dataset(args.train_dataset, split=args.train_dataset_split)
    train_dataset = train_dataset.map(
        lambda example, idx: {"indices": idx},
        with_indices=True, num_proc=10
    )


    test_dataset = load_dataset(args.test_dataset, split=args.test_dataset_split)
    test_dataset = test_dataset.map(
        lambda example, idx: {"indices": idx},
        with_indices=True, num_proc=10
    )
    
    estimators = [
        LESSEstimator(args.model_dir, train_dataset, args.train_dataset, args.train_dataset_split, test_dataset, args.test_dataset, args.test_dataset_split),
        LESSEstimator(args.model_dir, train_dataset, args.train_dataset, args.train_dataset_split, test_dataset, args.test_dataset, args.test_dataset_split, normalize=False),
        DataInfEstimator(args.model_dir, train_dataset, args.train_dataset, args.train_dataset_split, test_dataset, args.test_dataset, args.test_dataset_split),
        BM25Estimator(args.model_dir, train_dataset, args.train_dataset, args.train_dataset_split, test_dataset, args.test_dataset, args.test_dataset_split)
    ]
    for estimator in estimators:
        print(estimator.get_config_string)
if __name__ == "__main__":
    main()
