# !/bin/bash

uv run python scripts/memmap_maker.py \
    --output_memmap_path='data/' \
    --dataset="Salesforce/wikitext" \
    --subset="wikitext-2-raw-v1" \
    --dataset_columns="text" \
    --tokenizer="FacebookAI/roberta-base" \
