from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

def prepare_dataset(tokenizer, max_length=128):
    dataset = load_dataset("sentence-transformers/stsb")
    
    def tokenize(batch):
        return tokenizer(batch['sentence1'], batch['sentence2'],
                         padding='max_length',
                         truncation=True,
                         max_length=max_length,
                         return_tensors='pt')
    
    dataset = dataset.map(lambda x: tokenize(x), batched=True)
    dataset = dataset.map(lambda x: {'labels': x['score']}, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Load STS-B dataset
dataset = prepare_dataset(tokenizer, max_length=256)
train_dl = DataLoader(dataset['train'], batch_size=2, shuffle=True)
# = DataLoader(dataset['validation'], batch_size=args.batch_size)

for batch in train_dl:
    print(batch)