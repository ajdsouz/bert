from transformers import AutoTokenizer
from datasets import load_dataset
import torch

from src.dataset.tokenized_memmap import tokenize_dataset_as_memmap
from src.dataset.tokenized_memmap import TokenizedMemmapDataset


TOKENIZER = "openai-community/gpt2"
DATASET = ["uonlp/CulturaX", "af"]
SPLIT = "train[:10]"
COLUMN = "text"
MEMMAP_PATH = "tmp/test.npmmap"


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
orig_dataset = load_dataset(*DATASET, split=SPLIT)

tokenize_dataset_as_memmap(MEMMAP_PATH, tokenizer, orig_dataset, COLUMN)
target_dataset = TokenizedMemmapDataset(MEMMAP_PATH)

assert len(orig_dataset) == len(target_dataset)
print(f"{len(orig_dataset)=} {len(target_dataset)=} with same size")

for idx in map(int, torch.randperm(len(orig_dataset))):
    x = tokenizer.encode(orig_dataset[COLUMN][idx], return_tensors="pt")[0]
    y = target_dataset[idx]
    assert (x == y).all()
    print(f"{idx=} {x.shape=} {y.shape=} are element-wise equal")

print("pass test")
