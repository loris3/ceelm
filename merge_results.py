import os
import argparse
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import subprocess

def copy_file(src_file, src_root, dst_root):
    rel_path = os.path.relpath(src_file, src_root)
    dst_file = os.path.join(dst_root, rel_path)
    if os.path.exists(dst_file):
        return
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    try:
        subprocess.run(["cp", "-p", src_file, dst_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error copying {src_file} to {dst_file}: {e}")

def write_batch(args):
    """Read a batch of Parquet files and write to a single intermediate file."""
    batch_files, batch_index, schema, tmp_dir = args
    batch_output = os.path.join(tmp_dir, f"batch_{batch_index}.parquet")
    broken_files = []

    with pq.ParquetWriter(batch_output, schema=schema) as writer:
        for file in batch_files:
            try:
                table = pq.read_table(file, schema=schema)
                writer.write_table(table)
            except Exception as e:
                print(f"Broken file: {file} ({e})")
                broken_files.append(file)

    return batch_output, broken_files

def merge_parquet_files(source_dir, target_dir, max_workers=None):
    os.makedirs(target_dir, exist_ok=True)
    tmp_dir = os.path.join("/tmp", "tmp_batches_p")
    os.makedirs(tmp_dir, exist_ok=True)

    files = [os.path.join(root, f)
             for root, _, filenames in os.walk(source_dir)
             for f in filenames if f.endswith(".parquet")]

    if not files:
        print("No parquet files found.")
        return

    schema = pq.ParquetFile(files[0]).schema_arrow

    if max_workers is None:
        max_workers = cpu_count()

    # Split files into batches
    def chunks(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

    batches = chunks(files, max_workers)
    batch_args = [(batch, i, schema, tmp_dir) for i, batch in enumerate(batches)]

    all_broken_files = []
    intermediate_files = []

    with Pool(processes=max_workers) as pool:
        for batch_output, broken_files in tqdm(pool.imap_unordered(write_batch, batch_args),
                                               total=len(batch_args),
                                               desc="Merging of batches"):
            intermediate_files.append(batch_output)
            all_broken_files.extend(broken_files)

    # Write final merged Parquet
    output_file = os.path.join(target_dir, "merged.parquet")
    with pq.ParquetWriter(output_file, schema=schema) as writer:
        for batch_file in tqdm(intermediate_files, desc="Writing final output"):
            table = pq.read_table(batch_file, schema=schema)
            writer.write_table(table)

    if all_broken_files:
        fail_list_file = os.path.join(f"merge_fail_list_{os.path.basename(target_dir)}.txt")
        with open(fail_list_file, "w") as f:
            for bf in all_broken_files:
                f.write(bf + "\n")
        print(f"There were broken files, see {fail_list_file}")

    print(f"Merged Parquet file saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge Parquet fragments into a single dataset.")
    parser.add_argument("--source_dir", required=True, type=str, help="Path to the source directory containing Parquet fragments.")
    parser.add_argument("--target_dir", required=True, type=str, help="Path to the target directory where the merged dataset will be saved.")
    args = parser.parse_args()

    max_workers = cpu_count()

    print(f"{args.source_dir} --> {args.target_dir}", flush=True)
    os.makedirs(args.source_dir, exist_ok=True)
    os.makedirs(args.target_dir, exist_ok=True)


    merge_parquet_files(args.source_dir, args.target_dir, max_workers=max_workers)

if __name__ == "__main__":
    main()
