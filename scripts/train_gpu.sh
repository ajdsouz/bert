# !/bin/bash
# Usage: ./train.sh <gpu_id>

GPU_ID=$1

if [ -z "$GPU_ID" ]; then
  echo "Usage: $0 <GPU_ID>"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

uv run scripts/train.py \
    --model="FacebookAI/roberta-base" \
    --memmap_path='data' \
    --batch_size=1 \
    --block_size=128 \
    --d_model=256 \
    --d_ffn=512 \
    --n_heads=8 \
    --n_layer=8 \
    --dropout=0.0 \
    --vocab_size=50265 \
    --lr=5e-5 \
    --checkpoint_dir='ckpt' \
    --log_file='logs/logfile.log' \
    --wandb_entity="tororo" \
    --wandb_project_name="BERT Pretraining GPU Test" \
    --model_compile=True \
    --device="cuda" \
    --grad_accumulation_steps=4 \
    --precision="mixed-16" \
    --tracking=True