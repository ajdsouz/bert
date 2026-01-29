import argparse
import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from bert.model import BertEncoder, BERTConfigTemplate

class CLSHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x, attention_mask):
        # x: [B, T, D], attention_mask: [B, T]
        mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1)  # [B, D]
        return self.linear(pooled).squeeze(-1)  # [B]


class CLSTrainer(L.LightningModule):
    def __init__(self, model, lr=3e-5):
        super().__init__()
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.valid_acc = torchmetrics.Accuracy(task="binary")

    def forward(self, input_ids, attention_mask):
        x = self.model.transformer.wte(input_ids)
        x = self.model.transformer.spe(x)
        for block in self.model.transformer.h:
            x = block(x, attention_mask)
        x = self.model.transformer.ln_f(x)
        logits = self.model.head(x, attention_mask)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'].float()
        preds = self(input_ids, attention_mask)
        loss = self.loss_fn(preds, labels)
        probs = torch.sigmoid(preds)

        self.train_acc.update(probs, labels.int())

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'].float()
        preds = self(input_ids, attention_mask)
        loss = self.loss_fn(preds, labels)
        probs = torch.sigmoid(preds)
        self.valid_acc.update(probs, labels.int())


        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.valid_acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# -------------------------------
# Dataset prep
# -------------------------------
def prepare_dataset(tokenizer, max_length=128):
    dataset = load_dataset("stanfordnlp/imdb")
    
    def tokenize(batch):
        return tokenizer(batch['text'],
                         padding='max_length',
                         truncation=True,
                         max_length=max_length,
                         return_tensors='pt')
    
    dataset = dataset.map(lambda x: tokenize(x), batched=True)
    dataset = dataset.map(lambda x: {'labels': x['label']}, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset

# -------------------------------
# Main
# -------------------------------
def main(args):
    print("Tokenizer prepared")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    

    # Load imdb dataset
    print("loading dataset")
    dataset = prepare_dataset(tokenizer, max_length=args.block_size)
    train_dl = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(dataset['validation'], batch_size=args.batch_size)
    print("loaded dataset")
    
    # Load pretrained model
    class CLSConfig(BERTConfigTemplate):
        block_size = args.block_size
        d_model = args.d_model
        d_ffn = args.d_ffn
        n_heads = args.n_heads
        n_layer = args.n_layer
        dropout = args.dropout
        vocab_size = args.vocab_size

    print("loading model state dict from checkpoint")
    model = BertEncoder(CLSConfig)
    ckpt = torch.load(f"{args.checkpoint_dir}/{args.checkpoint_path}", map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print("loaded model checkpoint")

    # Replace LM head with regression head
    print("replacing lm_head with regressor head")
    model.head = CLSHead(args.d_model)

    print("freezing model weights except regressor head")
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    # Lightning module
    plmodel = CLSTrainer(model, lr=args.lr)

    # Logger

    wandb_logger = WandbLogger(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        name=args.run_name
    )

    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        accelerator=args.device,
        precision=args.precision,
        logger=wandb_logger,
        log_every_n_steps=1
    )

    # Train / evaluate
    trainer.fit(plmodel, train_dl, val_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--d_model', type=int, required=True)
    parser.add_argument('--d_ffn', type=int, required=True)
    parser.add_argument('--n_heads', type=int, required=True)
    parser.add_argument('--n_layer', type=int, required=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints")
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_project_name', type=str, default="sts-benchmark")
    parser.add_argument('--run_name', type=str, default="stsb_run")
    parser.add_argument('--device', type=str, default="gpu")
    parser.add_argument('--precision', type=str, default="fp32")
    args = parser.parse_args()
    main(args)
