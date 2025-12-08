import wandb
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger

class ModelTrainer(L.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            loss_fn: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        """Trainer class

        Args:
            checkpoint_dir (str): directory to save checkpoints to
            log_file (str): file save training logs
            tracking (bool): Tracking experiments with wandb
            wandb_project_name (str | None): wandb project name
            wandb_entity (str | None): wandb username
            model (nn.Module): Pytorch model
            loss_fn (nn.Module): loss function
            optimizer (torch.optim.Optimizer): optimizer
            scheduler (torch.optim.lr_scheduler.LRScheduler | None, optional): Learning rate scheduler. Defaults to None.
            compile (bool, optional): Compile the model?. Defaults to True.
            device (str | None, optional): device to use for training. Defaults to "cuda".
        """
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.tokens_seen = 0
        parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model has {parameters} parameters")

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx) -> torch.Tensor:
        """Computes masked language modelling forward pass

        Args:.to(self.device)
            batch (dict[str, torch.Tensor]): batch of data

        Returns:
            torch.Tensor: loss tensor
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels =batch['labels']
        labels = torch.where((labels >= 0) & (labels < self.model.config.vocab_size), labels, -100)
        outputs = self.model(input_ids, attention_mask)
        loss = self.loss_fn(
            outputs.view(-1, outputs.size(-1)), labels.view(-1)
        )
        tokens = attention_mask.sum()
        self.tokens_seen += tokens
        self.log("tokens_this_step", self.tokens_seen, on_step=True, on_epoch=False)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss
    

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx) -> None:
        """Evaluates model on validation dataset

        Args:
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader

        Returns:
            float: total loss for all batches in validation dataset
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels =batch['labels']
        labels = torch.where((labels >= 0) & (labels < self.model.config.vocab_size), labels, -100)
        outputs = self.model(input_ids, attention_mask)
        loss = self.loss_fn(
            outputs.view(-1, outputs.size(-1)), labels.view(-1)
        )
        self.log('val_loss', loss, prog_bar=True)


    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self._scheduler is not None:
            return {
                "optimizer": self._optimizer,
                "lr_scheduler": {
                    "scheduler": self._scheduler,
                    "monitor": "val_loss",
                }
            }
        else:
            return self._optimizer