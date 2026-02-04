from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, AutoTokenizer


from bert.dataset import TokenDataset, TokenDatasetV2

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
collate_fn = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm=True,
    mlm_probability=0.15
)

BOS_TOKEN_ID = tokenizer.bos_token_id
EOS_TOKEN_ID = tokenizer.eos_token_id
valid_ds = TokenDataset(memmap_path='test_data/validation.tokens', block_size=64, num_tokens=None)

valid_dl = DataLoader(valid_ds, batch_size=2, shuffle=True, pin_memory=True, collate_fn=collate_fn)

for sample in valid_dl:
    for key in sample.keys():
        print(f"{key}: {sample[key]}")
        print(tokenizer.batch_decode(sample['input_ids']))
    break

valid2_ds = TokenDatasetV2(memmap_path='test_data/validation.tokens', block_size=64, num_tokens=None, bos_token_id=BOS_TOKEN_ID, eos_token_id=EOS_TOKEN_ID)

valid2_dl = DataLoader(valid2_ds, batch_size=2, shuffle=True, pin_memory=True, collate_fn=collate_fn)

for sample in valid2_dl:
    for key in sample.keys():
        print(f"{key}: {sample[key]}")
        print(tokenizer.batch_decode(sample['input_ids']))
    break