import os
import argparse
from pyarrow import dataset as ds
import pyarrow.parquet as pq


def merge_fragments_with_source_column_parallel(source_dir, target_dir):
    """Merge Parquet fragments from a source directory into a target directory."""
    if True or not os.path.exists(target_dir):
        print(f"Loading {source_dir}")
        tmp_dataset = ds.dataset(source_dir, format="parquet")
        print(f"Merging {source_dir}")
        os.makedirs(target_dir, exist_ok=True)
        ds.write_dataset(
            tmp_dataset,
            target_dir,
            format="parquet",
            min_rows_per_group=25_000,
            max_rows_per_file=5_000_000,
            existing_data_behavior="overwrite_or_ignore"
        )
        print(f"Saved to {target_dir}")
    else:
        print(f"{target_dir} exists, skipping")


def main():
    parser = argparse.ArgumentParser(
        description="Merge Parquet fragments into a single dataset."
    )

    parser.add_argument(
        "--source_dir",
        required=True,
        type=str,
        help="Path to the source directory containing Parquet fragments."
    )

    parser.add_argument(
        "--target_dir",
        required=True,
        type=str,
        help="Path to the target directory where the merged dataset will be saved."
    )

    args = parser.parse_args()
    merge_fragments_with_source_column_parallel(args.source_dir, args.target_dir)


if __name__ == "__main__":
    main()
