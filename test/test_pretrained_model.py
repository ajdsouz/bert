import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from src.modeling import load_model, counting_parameters
from src.unified_dataset import load_unified_dataset
from src.utils import make_config_from_dict


DATASET_ROOT = "./data/"
PRETRAINED_PATH = "./results/{model_name}/checkpoint-65536"


dataset_config = make_config_from_dict({
    "type": "MemmapTokenDataset",
    "name": "memmap_token_open-web-text",
    "block_size": 512,
    "num_tokens": {"train": 510*42, "validation": 510*24},
    "tokenizer": "FacebookAI/roberta-base",
    "memmap_dtype": "uint16",
}) # fmt:skip

model_config = make_config_from_dict({
    "type": "MyBertModelForMlm",
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
    "tie_word_embeddings": True,
}) # fmt:skip

hf_model_config = make_config_from_dict({
    "type": "HuggingfaceModelForMlm",
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
assert torch.all(model.base.token_embedding.weight.data == model.mlm_head.weight.data)

print(f"""
## Model
{model}

### {model_config.type}
Model device:           {model.device}
Model dtype:            {model.dtype}
Number of Parameters:   {counting_parameters(model)[0]:,}

### {hf_model_config.type}
Model device:           {hf_model.device}
Model dtype:            {hf_model.dtype}
Number of Parameters:   {counting_parameters(hf_model)[0]:,}
""")# fmt:skip

dataset = load_unified_dataset(dataset_config, DATASET_ROOT, ("validation",))["validation"] # fmt: skip
data = dataset[0]
assert len(dataset) == 24
assert data.keys() == {"input_ids"}
assert (*data["input_ids"].shape,) == (dataset_config.block_size,)

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH.format(model_name="mybert-l8-h512-a8_mlm")) # fmt:skip
assert tokenizer.model_max_length == model_config.base_config.max_sequence_length
assert tokenizer.sep_token_id + 1 == len(tokenizer) == model_config.base_config.vocab_size
collate_fn = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)
inputs = collate_fn([dataset[i]["input_ids"] for i in range(2)])
assert "attention_mask" not in inputs
inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

outputs = model(**inputs)
hf_outputs = hf_model(**inputs)
print(outputs.loss)
print(hf_outputs.loss)
assert 0 <= outputs.loss <= 3
assert 0 <= hf_outputs.loss <= 3
