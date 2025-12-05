from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, AutoTokenizer


from bert.dataset import TokenDataset

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
collate_fn = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm=True,
    mlm_probability=0.15
)

valid_ds = TokenDataset(memmap_path='data/validation.tokens', block_size=64, num_tokens=1_000_000)

valid_dl = DataLoader(valid_ds, batch_size=2, shuffle=True, pin_memory=True, collate_fn=collate_fn)

for sample in valid_dl:
    for key in sample.keys():
        print(f"{key}: {sample[key]}")
        # print(tokenizer.batch_decode(sample['input_ids']))
    break