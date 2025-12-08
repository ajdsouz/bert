from bert.model import BertEncoder, BERTConfigTemplate
from bert.dataset import TokenDataset
from bert.trainer import Trainer

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, AutoTokenizer

def main(args) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collate_fn = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    train_ds = TokenDataset(memmap_path=f"{args.memmap_path}/train.tokens", block_size=args.block_size, num_tokens=args.num_tokens)
    valid_ds = TokenDataset(memmap_path=f"{args.memmap_path}/train.tokens", block_size=args.block_size, num_tokens=None)

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
        tracking=args.tracking,
        wandb_entity=args.wandb_entity,
        wandb_project_name=args.wandb_project_name,
        compile=args.model_compile,
        device=args.device,
        precision=args.precision,
    )

    trainer.train(
        train_dataloader=train_dl,
        val_dataloader=valid_dl,
        num_epochs=args.num_epochs,
        eval_every=args.eval_every,
        save_every=args.save_every,
        grad_accumulation_steps=args.grad_accumulation_steps
    )


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Pretraining BERT")
    parser.add_argument('--model', help="MODEL NAME",type=str)
    parser.add_argument('--memmap_path', help="MEMMAP PATH", type=str)
    parser.add_argument('--batch_size', help="BATCH SIZE", type=int)
    parser.add_argument('--block_size', help="BLOCK SIZE", type=int)
    parser.add_argument('--d_model', help="D_MODEL", type=int)
    parser.add_argument('--d_ffn', help="D_FFN",type=int)
    parser.add_argument('--n_heads', help="N_HEADS", type=int)
    parser.add_argument('--n_layer', help="N_LAYERS", type=int)
    parser.add_argument('--dropout', help="DROPOUT",type=float)
    parser.add_argument('--vocab_size', help="VOCAB_SIZE", type=int)
    parser.add_argument('--lr', help="LEARNING_RATE", type=float)
    parser.add_argument('--num_epochs', help="NUMBER OF EPOCHS", type=int)
    parser.add_argument('--eval_every', help="RUN EVAL AFTER EVERY n STEPS", type=int)
    parser.add_argument('--save_every', help="CHECKPOINT AFTER EVERY n STEPS", type=int)
    parser.add_argument('--checkpoint_dir', help="CHECKPOINT_DIR", type=str)
    parser.add_argument('--log_file', help="TRAINING LOGFILE PATH", type=str)
    parser.add_argument('--tracking', help="USE WANDB TO TRACK EXPERIMENTS?", type=str)
    parser.add_argument('--wandb_entity', help="WANDB USERNAME", type=str)
    parser.add_argument('--wandb_project_name', help="WANDB PROJECT NAME", type=str)
    parser.add_argument('--model_compile', help="COMPILE MODEL?", type=bool)
    parser.add_argument('--device', help="TRAINING ACCELERATOR", type=str)
    parser.add_argument('--precision', help="TRAINING PRECISION: mixed-16, bf16 or fp32", type=str)
    parser.add_argument('--grad_accumulation_steps', help="GRAD ACCUMULATION STEPS", type=int)
    parser.add_argument('--num_tokens', help="NO. OF TOKENS TO TRAIN MODEL ON",type=int)
    args = parser.parse_args()
    main(args)
