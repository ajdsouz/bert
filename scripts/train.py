from bert.model import BertEncoder, BERTConfigTemplate
from bert.dataset import TokenDataset
from bert.trainer import Trainer

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--memmap_path', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--block_size', type=int)
parser.add_argument('--d_model', type=int)
parser.add_argument('--d_ffn', type=int)
parser.add_argument('--n_heads', type=int)
parser.add_argument('--n_layer', type=int)
parser.add_argument('--dropout', type=int)
parser.add_argument('--vocab_size', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--checkpoint_dir', type=str)
parser.add_argument('--log_file', type=str)
parser.add_argument('--wandb_entity', type=str)
parser.add_argument('--wandb_project_name', type=str)
parser.add_argument('--model_compile', type=bool)
parser.add_argument('--device', type=str)
parser.add_argument('--grad_accumulation_steps', type=int)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
collate_fn = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm=True,
    mlm_probability=0.15
)

train_ds = TokenDataset(memmap_path=f"{args.memmap_path}/train.tokens", block_size=args.block_size)
valid_ds = TokenDataset(memmap_path=f"{args.memmap_path}/train.tokens", block_size=args.block_size)

train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4, collate_fn=collate_fn)
valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4, collate_fn=collate_fn)

class BERTTestConfig(BERTConfigTemplate):
    block_size: int = args.block_size
    d_model: int = args.d_model
    d_ffn: int = args.d_ffn
    n_heads: int = args.n_heads
    n_layer: int = args.n_layer
    dropout: float = args.dropout
    vocab_size: int = args.vocab_size

model = BertEncoder(BERTTestConfig)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
trainer = Trainer(
    model = model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    checkpoint_dir=args.checkpoint_dir,
    log_file=args.log_file,
    wandb_entity=args.wandb_entity,
    wandb_project_name=args.wandb_project_name,
    compile=args.model_compile,
    device=args.device
)

trainer.train(
    train_dataloader=train_dl,
    val_dataloader=valid_dl,
    grad_accumulation_steps=args.grad_accumulation_steps
)

