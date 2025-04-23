import os
import requests
from datasets import load_dataset
from transformers import AutoTokenizer # Keep this from previous step
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Make sure you are logged in to Hugging Face Hub and accepted Gemma terms
# You might need `from huggingface_hub import login` and `login()`
tokenizer_id = "google/gemma-2b" # Or another Gemma model ID
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

# download the tiny shakespeare dataset
#input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
#if not os.path.exists(input_file_path):
#    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#    with open(input_file_path, 'w', encoding='utf-8') as f:
#        f.write(requests.get(data_url).text)


# Load the specific Swedish subset from Hugging Face
# Use split='train' or specify train/validation splits as needed
# You might want a smaller subset initially for testing, e.g., dataset = load_dataset("HPLT/HPLT2.0_cleaned", "swe_Latn", split='train[:1%]')
dataset = load_dataset("HPLT/HPLT2.0_cleaned", "swe_Latn", split='train') 
# Example: Split it into train/val if only 'train' is loaded (adjust test_size as needed)
split_dataset = dataset.train_test_split(test_size=0.005, seed=42, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # Rename test to val


# Define the tokenization function (using Gemma tokenizer)
def process(example):
    # Assuming the text is in a column named 'text'. Check the dataset structure on HF if needed.
    ids = tokenizer.encode(example['text']) 
    # Add EOS token if desired/needed by nanoGPT structure? Gemma tokenizer might handle this.
    # ids.append(tokenizer.eos_token_id) # Check if Gemma tokenizer adds this automatically or if nanoGPT needs it.
    out = {'ids': ids, 'len': len(ids)}
    return out

# Tokenize the dataset (this can take time and disk space for caching)
tokenized = split_dataset.map(
    process,
    remove_columns=['text'], # Remove the original text column
    desc="tokenizing the splits",
    num_proc=os.cpu_count() // 2, # Use multiple cores if available
)

# Concatenate all IDs and save to bin files (similar logic to openwebtext/prepare.py)
data_dir = os.path.dirname(__file__)
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(data_dir, f'{split}.bin')
    dtype = np.uint16 # Check if Gemma token IDs fit in uint16 (vocab size 256k suggests yes)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024 # Or adjust based on dataset size/memory

    idx = 0
    # TQDM is useful here
    from tqdm import tqdm 
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()
    print(f"{split} has {arr_len:,} tokens")

# You don't need meta.pkl for this BPE tokenizer setup
