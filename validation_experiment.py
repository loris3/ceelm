import os
import random
import torch
import pandas as pd
from datasets import load_dataset
from validation_engine import ValidationEngine
from load_experiment_data import (
    load_data_and_estimators,
    train_dataset_name,
    test_dataset_name,
    train_dataset_split,
    test_dataset_split,
)
from explanations import Self, KRandom


torch.manual_seed(42)
random.seed(42)

train_dataset, test_dataset, estimators = load_data_and_estimators()
estimator = estimators[1] 
print("Using estimator:", estimator.model_path)


ood_ds = load_dataset("allenai/tulu-3-sft-personas-instruction-following")["train"]
ood_ds = ood_ds.map(
    lambda example: {"messages": example["messages"]},
    remove_columns=[col for col in ood_ds.column_names if col != "messages"]
)



lrs = [1e-5, 1e-4,]#1e-6, ]
epochs_list = [1,]#2,8,16,32]
repeats = [1,]#2,8,16,32]

indices = list(random.sample(range(len(test_dataset)), 100))#[0:10]
print("Total indices:", len(indices))


parquet_file = "cache/validation_of_validation/deltas_real.parquet"
os.makedirs(os.path.dirname(parquet_file), exist_ok=True)

# Load or create DataFrame to store results
if os.path.exists(parquet_file):
    df = pd.read_parquet(parquet_file)
else:
    df = pd.DataFrame(columns=[
        "LR", "Epoch", "Repeat", "Index", "Kind",
        "Delta", "log_p_before", "log_p_after", "JSD", "KL"
    ])


for lr in lrs:
    for epoch in epochs_list:
        engine = ValidationEngine(estimator.model_path, lr=lr, epochs=epoch)

        for repeat in repeats:
            print(f"=== LR={lr}, Epochs={epoch}, Repeat={repeat} ===")
            torch.manual_seed(42)
            random.seed(42)

            for idx in indices:
                if ((df["LR"] == lr) & (df["Epoch"] == epoch) &
                    (df["Index"] == idx) & (df["Repeat"] == repeat)).any():
                    continue


             
                test_examples = test_dataset.select([idx])
                
                explanation = KRandom(idx, estimator, k=1)
                train_examples = train_dataset.select(explanation.documents * repeat)
                metrics_random = engine.score(train_examples, test_examples, seed=42)
                
                train_examples = test_dataset.select([idx] * repeat)
                print("train_examples", train_examples["messages"])
                print("test_examples", test_examples["messages"])
                assert train_examples[0] == test_examples[0]
                metrics_exact = engine.score(test_examples, test_examples, seed=42)


                train_examples = ood_ds.select([idx] * repeat)
                metrics_ood = engine.score(train_examples, test_examples, seed=42)
                exact_docs = Self(idx, estimator, k=1).documents
                random_docs = KRandom(idx, estimator, k=1, seed=42).documents
                print("Exact:", exact_docs, "Random:", random_docs)

                new_rows = pd.DataFrame([
                    {
                        "LR": lr, "Epoch": epoch, "Index": idx, "Repeat": repeat,
                        "Kind": "exact",
                        "Delta": metrics_exact["delta_log_p"].mean().item(),
                        "log_p_before": metrics_exact["log_p_before_ft"].mean().item(),
                        "log_p_after": metrics_exact["log_p_after_ft"].mean().item(),
                        "JSD": metrics_exact["jsd"].mean().item(),
                        "KL": metrics_exact["kl(before||after)"].mean().item()
                    },
                    {
                        "LR": lr, "Epoch": epoch, "Index": idx, "Repeat": repeat,
                        "Kind": "random",
                        "Delta": metrics_random["delta_log_p"].mean().item(),
                        "log_p_before": metrics_random["log_p_before_ft"].mean().item(),
                        "log_p_after": metrics_random["log_p_after_ft"].mean().item(),
                        "JSD": metrics_random["jsd"].mean().item(),
                        "KL": metrics_random["kl(before||after)"].mean().item()
                    },
                    {
                        "LR": lr, "Epoch": epoch, "Index": idx, "Repeat": repeat,
                        "Kind": "ood",
                        "Delta": metrics_ood["delta_log_p"].mean().item(),
                        "log_p_before": metrics_ood["log_p_before_ft"].mean().item(),
                        "log_p_after": metrics_ood["log_p_after_ft"].mean().item(),
                        "JSD": metrics_ood["jsd"].mean().item(),
                        "KL": metrics_ood["kl(before||after)"].mean().item()
                    },
                ])
                df = pd.concat([df, new_rows], ignore_index=True)

                print(
                   
                    "Exact >= Random?", metrics_exact["delta_log_p"].mean().item() >= metrics_random["delta_log_p"].mean().item(),
                    "KL >= Random?", metrics_exact["kl(before||after)"].mean().item() >= metrics_random["kl(before||after)"].mean().item(),
                    "JSD >= Random?", metrics_exact["jsd"].mean().item() >= metrics_random["jsd"].mean().item(),
                     "metrics_exact:", metrics_exact,
                    "metrics_random:", metrics_random,
                    "metrics_ood:", metrics_ood,  

                )



                df.to_parquet(parquet_file, index=False)

        torch.cuda.empty_cache()
