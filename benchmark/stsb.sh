GPU_ID=$1

if [ -z "$GPU_ID" ]; then
  echo "Usage: $0 <GPU_ID>"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID \
uv run --model_name="answerdotai/ModernBERT-base" \
--checkpoint_path='data' \
--block_size=128 \
--d_model=256 \
--d_ffn=512 \
--n_heads=4 \
--n_layer=2 \
--dropout=0.0 \
--vocab_size=50265 \
--lr=3e-5 \
--num_epochs=10 \
--checkpoint_dir='ckpt' \
--wandb_entity="tororo" \
--wandb_project_name="STS-B Benchmark" \
--run_name="baseline" \
--device="device" \
--precision='mixed-16'