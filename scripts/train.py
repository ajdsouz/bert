from bert.model import BertEncoder, ModelConfig
from bert.dataset import TokenDataset, TokenDatasetV2
from bert.trainer import Trainer

import argparse
import dataclasses
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--memmap_path', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--block_size', type=int)
parser.add_argument('--d_model', type=int)
parser.add_argument('--d_ffn', type=int)
parser.add_argument('--n_heads', type=int)
parser.add_argument('--n_layer', type=int)
parser.add_argument('--attention_dropout', type=float)
parser.add_argument('--hidden_dropout', type=float)
parser.add_argument('--layernorm_eps', type=float)
parser.add_argument('--vocab_size', type=int)
parser.add_argument('--activation', type=str)
parser.add_argument('--mlp_type', type=str)
parser.add_argument('--norm_position', type=str)
parser.add_argument('--lr', type=float)
parser.add_argument('--beta1', type=float)
parser.add_argument('--beta2', type=float)
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--checkpoint_dir', type=str)
parser.add_argument('--log_file', type=str)
parser.add_argument('--wandb_entity', type=str)
parser.add_argument('--wandb_project_name', type=str)
parser.add_argument('--wandb_run_name', type=str)
# parser.add_argument('--model_compile', type=str)
parser.add_argument('--device', type=str)
parser.add_argument('--grad_accumulation_steps', type=int)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--save_every', type=int)
parser.add_argument('--eval_every', type=int)
parser.add_argument('--num_train_tokens', type=int)
parser.add_argument('--num_val_tokens', type=int)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
collate_fn = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm=True,
    mlm_probability=0.15
)

BOS_TOKEN_ID = tokenizer.bos_token_id
EOS_TOKEN_ID = tokenizer.eos_token_id

train_ds = TokenDatasetV2(memmap_path=f"{args.memmap_path}/train.tokens", block_size=args.block_size, num_tokens=args.num_train_tokens, bos_token_id=BOS_TOKEN_ID, eos_token_id=EOS_TOKEN_ID)
valid_ds = TokenDatasetV2(memmap_path=f"{args.memmap_path}/validation.tokens", block_size=args.block_size, num_tokens=args.num_val_tokens, bos_token_id=BOS_TOKEN_ID, eos_token_id=EOS_TOKEN_ID)

# train_ds = TokenDataset(memmap_path=f"{args.memmap_path}/train.tokens", block_size=args.block_size, num_tokens=args.num_train_tokens)
# valid_ds = TokenDataset(memmap_path=f"{args.memmap_path}/validation.tokens", block_size=args.block_size, num_tokens=args.num_val_tokens)

train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0, collate_fn=collate_fn)
valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0, collate_fn=collate_fn)

"""class BERTTestConfig(BERTConfigTemplate):
    block_size: int = args.block_size
    d_model: int = args.d_model
    d_ffn: int = args.d_ffn
    n_heads: int = args.n_heads
    n_layer: int = args.n_layer
    dropout: float = args.dropout
    vocab_size: int = args.vocab_size"""

bertconfig = ModelConfig(
    block_size=args.block_size,
    d_model=args.d_model,
    d_ffn=args.d_ffn,
    n_heads=args.n_heads,
    n_layer=args.n_layer,
    attention_dropout=args.attention_dropout,
    hidden_dropout=args.hidden_dropout,
    layernorm_eps=args.layernorm_eps,
    vocab_size=args.vocab_size,
    activation=args.activation,
    mlp_type=args.mlp_type,
    norm_position=args.norm_position
)

print(dataclasses.asdict(bertconfig))


TOTAL_OPTIMIZER_STEPS = math.ceil((len(train_dl) / args.grad_accumulation_steps) * args.num_epochs)
WARMUP_STEPS = max(1, int(0.05 * TOTAL_OPTIMIZER_STEPS))

model = BertEncoder(bertconfig)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=TOTAL_OPTIMIZER_STEPS
)

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
trainer = Trainer(
    config=bertconfig,
    model = model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    checkpoint_dir=args.checkpoint_dir,
    log_file=args.log_file,
    wandb_entity=args.wandb_entity,
    wandb_project_name=args.wandb_project_name,
    wandb_run_name=args.wandb_run_name,
    # compile=args.model_compile,
    device=args.device
)

trainer.train(
    train_dataloader=train_dl,
    val_dataloader=valid_dl,
    grad_accumulation_steps=args.grad_accumulation_steps,
    num_epochs=args.num_epochs,
    eval_every=args.eval_every,
    save_every=args.save_every
)

