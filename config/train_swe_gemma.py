dataset = 'swedish'
vocab_size = 256000
out_dir = 'out-swedish-gemma'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'jotest-bpe-gemma2'
wandb_run_name = 'jotest3-mini-gpt-bpe-gemma2'

dataset = 'swedish'
gradient_accumulation_steps = 8
batch_size = 12
block_size = 1024 # context of up to 1024 previous characters

# baby GPT model :)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2

#learning_rate = 1e-3 # with baby networks can afford to go a bit higher
learning_rate = 6e-4
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

dtype = 'bfloat16'

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = True # when on DGX, PyTorch 2.0+ compilation offers significant speedups on A100s.
# launch tjis when on dgx:
# torchrun --standalone --nproc_per_node=8 train.py config/train_swe_gemma.py
# CUDA_VISIBLE_DEVICES=6 python train.py config/train_swe_gemma.py --device=cuda [any other args]
