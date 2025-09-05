import argparse
import os
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi

parser = argparse.ArgumentParser(description="Creates a custom split of an HF dataset.")

parser.add_argument("--source_dataset", type=str, help="Name of the source HF dataset (e.g., imdb)", default="allenai/tulu-v2-sft-mixture")
parser.add_argument("--target_dataset", type=str, help="Name of the target HF dataset", default="tulu-v2-sft-mixture")
parser.add_argument("--train_frac", type=float, default=0.1, help="Fraction of training set to keep")
parser.add_argument("--test_frac", type=float, default=0.01, help="Fraction of test set to keep")
parser.add_argument("--seed", type=int, default=42, help="Seed to use")
parser.add_argument("--private", action="store_true", help="Make the dataset private on HF Hub", default=True)
args = parser.parse_args()


dataset = load_dataset(args.source_dataset)


custom_split = {"train": args.train_frac, "test": args.test_frac}
new_dataset = DatasetDict()
for split_name, fraction in custom_split.items():
    actual_split = split_name if split_name in dataset else "train"
    num_samples = int(len(dataset[actual_split]) * fraction)
    new_dataset[split_name] = dataset[actual_split].shuffle(seed=args.seed).select(range(num_samples))

print(new_dataset)


new_dataset.push_to_hub(
    repo_id=args.target_dataset,
    private=args.private, 

)


dataset_card = f"""
This dataset is a subset of [{args.source_dataset}](https://huggingface.co/datasets/{args.source_dataset}).


## Dataset Splits

- Train fraction: {args.train_frac}
- Test fraction: {args.test_frac}
- Seed: {args.seed}

`new_dataset[split_name] = dataset[split_name].shuffle(seed=args.seed).select(range(int(len(dataset[split]) * fraction)))`

## Dataset Size

- Train samples: {len(new_dataset['train'])}
- Test samples: {len(new_dataset['test'])}

## License

See original dataset
"""






with open("README.md_tmp", "w") as f:
    f.write(dataset_card)


from huggingface_hub import HfApi

api = HfApi()


repo_id = (
    api.whoami()["name"] + "/" + args.target_dataset
    if "/" not in args.target_dataset
    else args.target_dataset
)


api.upload_file(
    path_or_fileobj="README.md_tmp", 
    path_in_repo="README.md", 
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Add dataset card"
)
os.remove("README.md_tmp")
