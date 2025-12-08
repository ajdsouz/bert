from bert.model import BertEncoder, BERTConfigTemplate
from bert.dataset import TokenDataset
from bert.trainer import ModelTrainer

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, AutoTokenizer

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


def main(args) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collate_fn = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    train_ds = TokenDataset(memmap_path=f"{args.memmap_path}/train.tokens", block_size=args.block_size, num_tokens=args.num_tokens)
    valid_ds = TokenDataset(memmap_path=f"{args.memmap_path}/validation.tokens", block_size=args.block_size, num_tokens=None)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4, collate_fn=collate_fn)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0, collate_fn=collate_fn)

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

    checkpoint_every_n_steps = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="epoch-{epoch}-step-{step}",                    
        every_n_train_steps=args.save_every,                                                                                                                                         
        save_top_k=-1,                             
        save_last=True,                            
    )

    checkpoint_best = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="best-{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        every_n_train_steps=0,      # disable step-based saving for this one
    )

    wandb_logger = WandbLogger(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        name=args.run_name
    )
    
    plmodel = ModelTrainer(
        model = model,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )

    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=args.num_epochs,
        precision=args.precision,
        accelerator=args.device,
        callbacks=[
            checkpoint_every_n_steps,
            checkpoint_best
        ],
        accumulate_grad_batches=args.grad_accumulation_steps,
        log_every_n_steps=1,
        val_check_interval=args.eval_every
    )
    

    trainer.fit(
        plmodel,
        train_dl,
        valid_dl,

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
    #parser.add_argument('--log_file', help="TRAINING LOGFILE PATH", type=str)
    #parser.add_argument('--tracking', help="USE WANDB TO TRACK EXPERIMENTS?", type=str)
    parser.add_argument('--wandb_entity', help="WANDB USERNAME", type=str)
    parser.add_argument('--wandb_project_name', help="WANDB PROJECT NAME", type=str)
    parser.add_argument('--run_name', help="WANDB RUN NAME", type=str)
    parser.add_argument('--device', help="TRAINING ACCELERATOR", type=str)
    parser.add_argument('--precision', help="TRAINING PRECISION: mixed-16, bf16 or fp32", type=str)
    parser.add_argument('--grad_accumulation_steps', help="GRAD ACCUMULATION STEPS", type=int)
    parser.add_argument('--num_tokens', help="NO. OF TOKENS TO TRAIN MODEL ON",type=int)
    args = parser.parse_args()
    main(args)
