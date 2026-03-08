from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('ckpt/tokenizer')

tokenizer.push_to_hub('ajdsouza/roberta-extended')

