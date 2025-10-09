import os
from pyarrow import dataset as ds

import pyarrow.parquet as pq

def merge_fragments_with_source_column_parallel(source_dir, target_dir):

    if True or not os.path.exists(target_dir):
        print(f"Loading {source_dir}" )
        tmp_dataset = ds.dataset(source_dir, format="parquet")
        print(f"Merging {source_dir}" )
        os.makedirs(target_dir, exist_ok=True)
        ds.write_dataset(
            tmp_dataset,
            target_dir,
            format="parquet",
            min_rows_per_group=25_000,
            max_rows_per_file=5_000_000,
        )
        print(f"Saved to {target_dir}" )
    else:
        print(f"{target_dir} exists, skipping" )
merge_fragments_with_source_column_parallel(source_dir="cache/validation/partial", target_dir="results/validation",)
# merge_fragments_with_source_column_parallel( source_dir="cache/scoring/partial", target_dir="results/scoring")
