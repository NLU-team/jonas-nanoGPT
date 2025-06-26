import os
import argparse
import concurrent.futures
from datasets import concatenate_datasets, Dataset
import numpy as np
from transformers import AutoTokenizer
from dotenv import load_dotenv
from tqdm import tqdm
tqdm(disable=True, total=0)

# Load environment variables from .env file
load_dotenv()

def get_parquet_files(dirs, dry_run=False):
    """Scans for Parquet files in the provided directories."""
    print("Scanning for Parquet files...")
    files = []
    for d in dirs:
        for root, _, filenames in os.walk(d):
            for f in filenames:
                if f.endswith(".parquet"):
                    files.append(os.path.join(root, f))
    if dry_run:
        files = files[:1]
    print(f"Found {len(files)} Parquet files.")
    return files

def load_parquet_dataset(file_path):
    """Loads a dataset from a single Parquet file."""
    print(f"Loading dataset from: {file_path}")
    # Using from_parquet on a single file path is correct
    return Dataset.from_parquet(file_path)

def remove_extra_columns(ds, columns_to_keep):
    """Removes columns not in the columns_to_keep list."""
    cols_to_remove = [col for col in ds.column_names if col not in columns_to_keep]
    if cols_to_remove:
        print(f"Removing columns: {cols_to_remove}")
        return ds.remove_columns(cols_to_remove)
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dirs", nargs="+", required=True, help="Directories containing your SWEB Parquet files.")
    parser.add_argument("--dry_run", action="store_true", help="Process a smaller subset for testing.")
    parser.add_argument("--cores", type=int, default=16, help="Number of cores for processing.")
    args = parser.parse_args()

    # --- 1. Load SWEB Datasets from Parquet files ---
    swb_datasets = []
    if args.parquet_dirs:
        parquet_files = get_parquet_files(args.parquet_dirs, dry_run=args.dry_run)
        if parquet_files:
            print("Loading SWEB Parquet datasets...")
            # Using map to load each file individually
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.cores) as executor:
                swb_datasets.extend(list(executor.map(load_parquet_dataset, parquet_files)))
            print(f"Loaded {len(swb_datasets)} SWEB datasets.")
    else:
        print("No SWEB Parquet directories provided. Exiting.")
        return

    # --- 2. Combine Datasets ---
    if not swb_datasets:
        print("No SWEB datasets loaded. Exiting.")
        return
        
    print("Processing and concatenating SWEB datasets...")
    processed_sweb = [remove_extra_columns(ds, ["text"]) for ds in swb_datasets]
    combined_ds = concatenate_datasets(processed_sweb)

    if args.dry_run:
        combined_ds = combined_ds.select(range(min(500, len(combined_ds))))
        print(f"Dataset reduced to {len(combined_ds)} for dry run.")

    # --- 3. Tokenization ---
    tokenizer_id = "data/swedish/adapted_gemma_nordic"  # Path to your custom tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    
    def process(example):
        ids = tokenizer.encode(example['text'])
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Split the combined dataset
    split_dataset = combined_ds.train_test_split(test_size=0.005, seed=42, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing the splits",
        num_proc=args.cores,
    )

    # --- 4. Save to Binary Files ---
    data_dir = os.path.dirname(__file__)
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(data_dir, f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        total_batches = 1024
        idx = 0
        from tqdm import tqdm
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"{split} has {arr_len:,} tokens")

if __name__ == "__main__":
    main()