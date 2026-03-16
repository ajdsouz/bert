from transformers import AutoTokenizer

from src.unified_dataset import load_unified_dataset
from src.utils import make_config_from_dict


ROOT = "./data/"

config = make_config_from_dict({
    "type": "HuggingfaceDataset",
    "name": "sst2",
    "path": "nyu-mll/glue",
    "subset_name": "sst2",
    "input_columns": ["sentence"],
    "target_columns": ["label"],
    "splits": {"train": "train", "validation": "validation"},
}) # fmt:skip

datasets = load_unified_dataset(config, ROOT, ("train", "validation"))
print(len(datasets["train"]))
print(len(datasets["validation"]))

data = datasets["train"][0]
assert data.keys() == {"inputs", "targets"}

for i in range(5):
    data = datasets["train"][i]
    inputs, targets = data["inputs"], data["targets"]
    print(f"item {i}:")
    print(f"  inputs : list[{type(inputs[0]).__name__}, {len(inputs)}]", inputs)
    print(f"  targets: list[{type(targets[0]).__name__}, {len(targets)}]", targets)
