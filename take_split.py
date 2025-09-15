import argparse
import os
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi

def parse_frac_or_num(value):
    try:
        f = float(value)
        if f <= 0:
            raise argparse.ArgumentTypeError("Value must be positive")
        return f
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}")

parser = argparse.ArgumentParser(description="Creates a custom split of an HF dataset. Accepts fractions (0-1) or absolute numbers for splits, e.g., 0.1 or 1000.")
parser.add_argument("--source_dataset", type=str, default="allenai/tulu-3-sft-olmo-2-mixture-0225")
parser.add_argument("--target_dataset", type=str, default="tulu-3-sft-olmo-2-mixture-0225-sample")
parser.add_argument("--train", type=parse_frac_or_num, default=0.1)
parser.add_argument("--test", type=parse_frac_or_num, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--private", action="store_true", default=True)
args = parser.parse_args()

dataset = load_dataset(args.source_dataset)

def get_num_samples(split, value):
    total = len(dataset[split])
    if 0 < value <= 1:
        return int(total * value)
    return min(int(value), total)

custom_split = {"train": args.train, "test": args.test}
new_dataset = DatasetDict()
for i, (split_name, value) in enumerate(custom_split.items()):
    actual_split = split_name if split_name in dataset else "train"
    num_samples = get_num_samples(actual_split, value)
    new_dataset[split_name] = dataset[actual_split].shuffle(seed=args.seed + i * 42).select(range(num_samples))

print(new_dataset)

new_dataset.push_to_hub(repo_id=args.target_dataset, private=args.private)

total_train = len(dataset['train'])
total_test = len(dataset['train'])

dataset_card = f"""
---
license: cc
tags:
- custom
- subset
---
This dataset is a subset of [{args.source_dataset}](https://huggingface.co/datasets/{args.source_dataset}).


### Generation Command

```bash
python take_split.py --source_dataset {args.source_dataset} --target_dataset {args.target_dataset} --train {args.train} --test {args.test} --seed {args.seed} {"--private" if args.private else ""}
```

## Dataset Splits

- Train: {len(new_dataset['train'])} samples ({len(new_dataset['train'])/total_train:.2%} of original)
- Test: {len(new_dataset['test'])} samples ({len(new_dataset['test'])/total_train:.2%} of original)
- Seed: {args.seed}
## License

See original dataset
"""

with open("README.md_tmp", "w") as f:
    f.write(dataset_card)

api = HfApi()
repo_id = api.whoami()["name"] + "/" + args.target_dataset if "/" not in args.target_dataset else args.target_dataset
api.upload_file(
    path_or_fileobj="README.md_tmp",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Add dataset card"
)
os.remove("README.md_tmp")