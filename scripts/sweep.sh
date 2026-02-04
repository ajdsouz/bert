# !/bin/bash
# Usage: ./sweep.sh <gpu_id>

GPU_ID=$1

if [ -z "$GPU_ID" ]; then
  echo "Usage: $0 <GPU_ID>"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
uv run scripts/sweep.py --lr=0.0001 \
    --memmap_path=/local/username/tokens \
    --checkpoint_base_dir=/local/username/sweeps/ckpts/ \
    --log_file_dir=/local/username/sweeps/logs/