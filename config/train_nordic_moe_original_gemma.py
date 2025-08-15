# config/train_nordic_moe_original_gemma.py

# I/O
out_dir = 'out-nordic-moe-small-original-gemma'
eval_interval = 1000
log_interval = 10
eval_iters = 200
always_save_checkpoint = True

# wandb logging
wandb_log = True
wandb_project = 'nordic-moe'
wandb_run_name = 'moe-small-original-gemma-tokenizer'

# data
dataset = 'swedish'  # This should be the directory containing your train.bin and val.bin
gradient_accumulation_steps = 20 * 5  # For 5 GPUs to simulate a larger batch size
batch_size = 2
block_size = 512
# Use the vocab size from the original Gemma tokenizer you used for the .bin files
vocab_size = 262144
init_from = 'resume' # 'resume' after crasch, 'scratch' from start

# model architecture for a ~1.7B MoE model
# model architecture (Reduced size due to cuda oom on dgx)
n_layer = 12       # Reduced from 18
n_head = 8         # Reduced from 16
n_embd = 1024      # Reduced from 2048
dropout = 0.0
bias = False

# MoE settings based on FLAME
n_experts = 8
n_experts_per_tok = 8
n_shared_experts = 2 # As per the FLAME paper's architecture

# adamw optimizer
learning_rate = 3e-4
max_iters = 600000  # Adjust as needed
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 3e-5 # learning_rate/10

# DDP settings
backend = 'nccl'

# system
device = 'cuda'
dtype = 'bfloat16'
compile = True