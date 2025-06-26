# train a 1.7B parameter MoE model on a custom Nordic dataset

# I/O
out_dir = 'out-nordic-moe-1.7b'
eval_interval = 1000
log_interval = 10
eval_iters = 200
always_save_checkpoint = True

# wandb logging
wandb_log = True
wandb_project = 'nordic-moe'
wandb_run_name = 'moe-1.7b'

# data
dataset = 'swedish'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12
block_size = 1024
vocab_size = 256000 # From your gemma tokenizer

# model
n_layer = 18
n_head = 16 # Adjusted for n_embd
n_embd = 2048
dropout = 0.0
bias = False

# MoE settings from FLAME paper
n_experts = 64
n_experts_per_tok = 8
n_shared_experts = 2

# adamw optimizer
learning_rate = 3e-4 # max learning rate
max_iters = 200000 # Adjust based on your dataset size and desired epochs
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 200000 # should be ~= max_iters
min_lr = 3e-5 # should be learning_rate/10

# DDP settings
backend = 'nccl'

# system
device = 'cuda'
dtype = 'bfloat16'
compile = True