import torch
from transformers import AutoTokenizer

from src.modeling import load_model, counting_parameters
from src.unified_dataset import load_unified_dataset
from src.utils import make_config_from_dict


DATASET_ROOT = "./data/"
PRETRAINED_PATH = "./results/{model_name}/checkpoint-65536"


dataset_config = make_config_from_dict({
    "type": "HuggingfaceDataset",
    "name": "sst2",
    "path": "nyu-mll/glue",
    "subset_name": "sst2",
    "input_columns": ["sentence"],
    "target_columns": ["label"],
    "splits": {"train": "train", "validation": "validation"},
}) # fmt:skip

model_config = make_config_from_dict({
    "type": "MyBertModelForSequenceClassification",
    "base_config": {
        "type": "MyBertModel",
        "device": "cpu",
        "dtype": "float32",
        "vocab_size": 50266,
        "max_sequence_length": 512,
        "output_hidden_states": False,
        "output_attentions": False,
        "hidden_size": 512,
        "num_encoder_layers": 8,
        "num_attention_heads": 8,
        "gated_attention": True,
        "mlp_type": "gated(gelu)",
    },
    "num_labels": 2,
}) # fmt:skip

hf_model_config = make_config_from_dict({
    "type": "HuggingfaceModelForSequenceClassification",
    "base_config": {
        "type": "HuggingfaceModel",
        "device": "cpu",
        "dtype": "float32",
        "vocab_size": 50266,
        "max_sequence_length": 512,
        "output_hidden_states": False,
        "output_attentions": False,
        "hf_repo": "google/bert_uncased_L-8_H-512_A-8",
    },
    "num_labels": 2,
}) # fmt:skip

model = load_model(
    model_config,
    pretrained_path=PRETRAINED_PATH.format(model_name="mybert-l8-h512-a8_mlm"),
)
hf_model = load_model(
    hf_model_config,
    pretrained_path=PRETRAINED_PATH.format(
        model_name="google_bert_uncased_L-8_H-512_A-8_mlm"
    ),
)

print(f"""

### {model_config.type}
{model}

Model device:           {model.device}
Model dtype:            {model.dtype}
Number of Parameters:   {counting_parameters(model)[0]:,}

### {hf_model_config.type}
{hf_model}

Model device:           {hf_model.device}
Model dtype:            {hf_model.dtype}
Number of Parameters:   {counting_parameters(hf_model)[0]:,}
""")# fmt:skip

dataset = load_unified_dataset(dataset_config, DATASET_ROOT, ("validation",))["validation"] # fmt:skip
data = dataset[0]
assert data.keys() == {"inputs", "targets"}
assert len(data["targets"]) == 1 and isinstance(data["targets"][0], int)

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH.format(model_name="mybert-l8-h512-a8_mlm")) # fmt:skip
assert tokenizer.model_max_length == model_config.base_config.max_sequence_length
assert tokenizer.sep_token_id + 1 == len(tokenizer) == model_config.base_config.vocab_size

inputs = tokenizer(tokenizer.sep_token.join(data["inputs"]), return_tensors="pt")
assert inputs["input_ids"][0][0] == tokenizer.bos_token_id
assert inputs["input_ids"][0][-1] == tokenizer.eos_token_id
labels = torch.tensor([data["targets"][0]])

outputs = model(**inputs, labels=labels)
hf_outputs = hf_model(**inputs, labels=labels)
print(outputs.loss)
print(hf_outputs.loss)
assert 0 < outputs.loss <= 1.5
assert 0 < hf_outputs.loss <= 1.5
