# !/bin/bash

uv run python scripts/memmap_maker.py \
    --output_memmap_path='test_data' \
    --dataset="radm/tathagata" \
    --subset=None \
    --dataset_columns="text" \
    --val_ratio=0.1 \
    --batch_size=8 \
    --num_proc=4 \
    --tokenizer="answerdotai/ModernBERT-base" \
