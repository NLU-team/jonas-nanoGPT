import os
import argparse
import concurrent.futures
from datasets import concatenate_datasets, Dataset
import numpy as np
from transformers import AutoTokenizer
from dotenv import load_dotenv
import pyarrow

# Define a temporary directory within the /data/ path where you have write permissions
temp_dir = '/data/tmp/pymp_jonas' # Or any other path in /data you can write to
os.environ['MULTIPROCESSING_FORK_DIR'] = temp_dir

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
    try:
        print(f"Loading dataset from: {file_path}")
        return Dataset.from_parquet(file_path)
    except pyarrow.lib.ArrowInvalid as e:
        print(f"!!!!!!!!!!!!! SKIPPING FILE DUE TO ArrowInvalid ERROR !!!!!!!!!!!!!")
        print(f"Error loading Parquet file {file_path}: {e}")
        print(f"This file might be empty or corrupted. Continuing with the next file.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None # Return None for problematic files
    except Exception as e:
        print(f"!!!!!!!!!!!!! SKIPPING FILE DUE TO UNEXPECTED ERROR !!!!!!!!!!!!!")
        print(f"Error loading Parquet file {file_path}: {e}")
        return None


def remove_extra_columns(ds, columns_to_keep):
    """Removes columns not in the columns_to_keep list."""
    cols_to_remove = [col for col in ds.column_names if col not in columns_to_keep]
    if cols_to_remove:
        print(f"Removing columns: {cols_to_remove}")
        return ds.remove_columns(cols_to_remove)
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dirs", nargs="+", required=True, help="Directories with SWEB Parquet files.")
    parser.add_argument("--dry_run", action="store_true", help="Process a smaller subset for testing.")
    parser.add_argument("--cores", type=int, default=16, help="Number of cores for processing.")
    args = parser.parse_args()

    # Create the temporary directory if it doesn't exist
    if 'MULTIPROCESSING_FORK_DIR' in os.environ:
        os.makedirs(os.environ['MULTIPROCESSING_FORK_DIR'], exist_ok=True)


    # --- 1. Load SWEB Datasets ---
    swb_datasets = []
    if args.parquet_dirs:
        parquet_files = get_parquet_files(args.parquet_dirs, dry_run=args.dry_run)
        if parquet_files:
            print(f"Loading SWEB Parquet datasets concurrently using temp dir: {temp_dir}...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.cores) as executor:
                # Use map to load each file individually and filter out None results from failed loads
                results = list(executor.map(load_parquet_dataset, parquet_files))
                swb_datasets = [ds for ds in results if ds is not None]
            print(f"Successfully loaded {len(swb_datasets)} SWEB datasets.")

    if not swb_datasets:
        print("No valid SWEB datasets were loaded. Exiting.")
        return

    print("Processing and concatenating SWEB datasets...")
    processed_sweb = [remove_extra_columns(ds, ["text"]) for ds in swb_datasets]
    combined_ds = concatenate_datasets(processed_sweb)

    if args.dry_run:
        combined_ds = combined_ds.select(range(min(500, len(combined_ds))))
        print(f"Dataset reduced to {len(combined_ds)} for dry run.")

    # --- 3. Tokenization ---
    tokenizer_id = "data/swedish/adapted_gemma_nordic" # Path to your custom tokenizer
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