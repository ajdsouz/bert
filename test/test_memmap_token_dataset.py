from transformers import AutoTokenizer

from src.unified_dataset import load_unified_dataset
from src.utils import make_config_from_dict


ROOT = "./data/"

config = make_config_from_dict({
    "type": "MemmapTokenDataset",
    "name": "memmap_token_open-web-text",
    "block_size": 512,
    "num_tokens": {"train": 0, "validation": 0},
    "tokenizer": "FacebookAI/roberta-base",
    "memmap_dtype": "uint16",
}) # fmt:skip

datasets = load_unified_dataset(config, ROOT, ("train", "validation"))
print(len(datasets["train"]))
assert len(datasets["train"]) == 3545246
assert len(datasets["train"]._memmap) == 1_808_075_825  # 1.8B
print(len(datasets["validation"]))
assert len(datasets["validation"]) == 3484
assert len(datasets["validation"]._memmap) == 1_777_110 # 1.7M

data = datasets["train"][0]
assert data.keys() == {"input_ids"}
assert (*data["input_ids"].shape,) == (config.block_size,)

# Add a unique sep token.
tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
assert tokenizer.eos_token_id == tokenizer.sep_token_id
tokenizer.add_special_tokens({"sep_token": "<sep>"})
assert tokenizer.sep_token_id + 1 == len(tokenizer) == 50266

sentence = tokenizer.decode(data["input_ids"])
print(data["input_ids"])
print(sentence)
