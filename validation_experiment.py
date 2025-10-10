import os
import random
import torch
import pandas as pd
import argparse
from datasets import load_dataset
from validation_engine import ValidationEngine
from load_experiment_data import load_data_and_estimators
from explanations import KRandom

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Validation of validation experiment runner")
parser.add_argument("--lr", type=float, required=True, help="Learning rate")
parser.add_argument("--epoch", type=int, required=True, help="Number of epochs")
parser.add_argument("--repeat", type=int, required=True, help="Repeat count for training examples")
args = parser.parse_args()

lr = args.lr
epoch = args.epoch
repeat = args.repeat


n_examples = 100
torch.manual_seed(42)
random.seed(42)

train_dataset, test_dataset, estimators = load_data_and_estimators()
estimator = estimators[1]  # just to get model

# out-of-distribution dataset (still tokenized using conversational format)
ood_ds = load_dataset("allenai/tulu-3-sft-personas-instruction-following")["train"]
ood_ds = ood_ds.map(
    lambda ex: {"messages": ex["messages"]},
    remove_columns=[c for c in ood_ds.column_names if c != "messages"],
)

indices = random.sample(range(len(test_dataset)), n_examples + 1)
print(f"Total indices: {len(indices)}")


parquet_folder = f"cache/validation_of_validation/{lr}/{epoch}/{repeat}"
os.makedirs(parquet_folder, exist_ok=True)

print(f"LR={lr}, Epochs={epoch}, Repeat={repeat}")

engine = ValidationEngine(estimator.model_path, lr=lr, epochs=epoch)

df = pd.DataFrame()


for idx in tqdm(indices):
    torch.manual_seed(42)
    random.seed(42)

    parquet_path = os.path.join(parquet_folder, f"{idx}.parquet")
    if os.path.exists(parquet_path):
        continue

    test_ex = test_dataset.select([idx])

    rand_docs = KRandom(idx, estimator, k=1).documents
    train_ex = train_dataset.select(rand_docs * repeat)
    metrics_random = engine.score(train_ex, test_ex, seed=42)

    train_ex = test_dataset.select([idx] * repeat)
    metrics_exact = engine.score(test_ex, test_ex, seed=42)

    train_ex = ood_ds.select([idx] * repeat)
    metrics_ood = engine.score(train_ex, test_ex, seed=42)

    print(f"Index {idx}: Exact vs Random vs OOD comparison")
    print(
        "Exact >= Random?",
        metrics_exact["delta_log_p"].mean().item() >= metrics_random["delta_log_p"].mean().item(),
        "| KL >= Random?",
        metrics_exact["kld(before||after)"].mean().item() >= metrics_random["kld(before||after)"].mean().item(),
        "| JSD >= Random?",
        metrics_exact["jsd"].mean().item() >= metrics_random["jsd"].mean().item(),
    )

    def make_row(kind, m):
        return {
            "LR": lr, "Epoch": epoch, "Repeat": repeat, "Index": idx, "Kind": kind,
            "delta_log_p": m["delta_log_p"].mean().item(),
            "log_p_before": m["log_p_before_ft"].mean().item(),
            "log_p_after": m["log_p_after_ft"].mean().item(),
            "jsd": m["jsd"].mean().item(),
            "kld(before||after)": m["kld(before||after)"].mean().item()
        }

    rows = [
        make_row("exact", metrics_exact),
        make_row("random", metrics_random),
        make_row("ood", metrics_ood),
    ]

    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_parquet(parquet_path, index=False)

    torch.cuda.empty_cache()
