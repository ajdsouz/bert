# !/bin/bash

uv run scripts/train.py \
    --model="answerdotai/ModernBERT-base" \
    --memmap_path='test_data' \
    --batch_size=1 \
    --block_size=128 \
    --d_model=256 \
    --d_ffn=512 \
    --n_heads=4 \
    --n_layer=2 \
    --dropout=0.0 \
    --vocab_size=50368 \
    --lr=5e-5 \
    --checkpoint_dir='ckpt' \
    --log_file='logs/logfile.log' \
    --wandb_entity="tororo" \
    --wandb_project_name="BERT Pretraining Test" \
    --wandb_run_name="custom trainer pure torch no mixed precision" \
    --model_compile=True\
    --device="cpu" \
    --grad_accumulation_steps=4 \
    --num_epochs=1 \
    --save_every=100 \
    --eval_every=50 \
    --num_train_tokens=1_000_000 \
    --num_val_tokens=1_000 \