from .model import ModelConfig

import wandb
import logging
import argparse
from tqdm import tqdm
import dataclasses
import torch
import torch.nn as nn

from lightning.fabric import Fabric

"""class Trainer:
    def __init__(
            self,
            config: ModelConfig,
            checkpoint_dir: str,
            log_file: str,
            wandb_project_name: str,
            wandb_entity: str,
            wandb_run_name: str,
            model: nn.Module,
            loss_fn,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
            compile: bool = True,
            device: str | None = "cuda",
    ) -> None:
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compile = compile
        self.checkpoint_dir = checkpoint_dir
        self.config = config
        self.epoch: int = 0
        self.global_step: int = 0

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
        formatter = logging.Formatter("{levelname:<8} {message}", style="{")
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        if self.compile:
            self.model = torch.compile(self.model)
            self.logger.info("Model is compiled!")

        wandb.init(
            entity=wandb_entity,
            project=wandb_project_name,
            name=wandb_run_name,
            config = dataclasses.asdict(config)
        )


    def _common_step(self, batch) -> torch.Tensor:
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels =batch['labels'].to(self.device)
        labels = torch.where((labels >= 0) & (labels < self.model.config.vocab_size), labels, -100)
        outputs = self.model(input_ids, attention_mask)
        loss = self.loss_fn(
            outputs.view(-1, outputs.size(-1)), labels.view(-1)
        )
        return loss


    def evaluate(self, val_dataloader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                loss = self._common_step(batch)
                total_loss += loss.item()

        return total_loss / len(val_dataloader)


    def train(
            self, 
            train_dataloader, 
            val_dataloader = None,
            num_epochs: int = 1,
            eval_every: int = 5000,
            save_every: int = 5000,
            grad_accumulation_steps: int = 1,
            grad_clip_max_norm: float = 1.0
    ) -> None:
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            total_loss = 0.0
            tokens_seen = 0
            epoch_steps = 0
            val_loss = None

            with tqdm(total=len(train_dataloader), desc=f"Training Epoch {self.epoch}", dynamic_ncols=True) as pbar:
                for idx, batch in enumerate(train_dataloader):

                    # --------- Training --------
                    # tokens_seen += (batch['input_ids'].size(0) * batch['input_ids'].size(1))
                    tokens_seen += batch['input_ids'].numel() # store no. of tokens model has seen in each batch
                    self.model.train()
                    
                    loss = self._common_step(batch)
                    loss_scaled = loss / grad_accumulation_steps
                    loss_scaled.backward()
                    

                    if ((idx + 1) % grad_accumulation_steps == 0) or (idx + 1 == len(train_dataloader)):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=grad_clip_max_norm
                        )
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.scheduler:
                            self.scheduler.step()

                    self.global_step += 1
                    epoch_steps += 1
                    total_loss += loss_scaled.item()


                    self.logger.info(f"Train epoch: {self.epoch} step: {self.global_step} Tokens seen: {tokens_seen} Train Loss: {loss.item()}")
                    wandb.log({'train_loss': loss.item(), 'tokens_seen': tokens_seen, 'epoch': self.epoch, 'step': self.global_step}, step=self.global_step)
                
                    # --------- evaluation ---------
                    if self.global_step % eval_every == 0 and val_dataloader and self.global_step > 0 or epoch_steps == len(train_dataloader):
                        val_loss = self.evaluate(val_dataloader)
                        self.logger.info(f"Eval Step: {self.global_step} Tokens seen: {tokens_seen} Val Loss : {val_loss}")
                        wandb.log({'val_loss': val_loss, 'tokens_seen': tokens_seen, 'epoch': self.epoch, 'step': self.global_step}, step=self.global_step)

                    # --------- checkpointing ---------
                    if self.global_step % save_every == 0:
                        self.save_checkpoint()
                        self.logger.info(f"Checkpoint Saved | Train loss: {loss} | Eval loss: {val_loss}")

                    pbar.set_description(f"Train Loss step : {loss.item():.4f} | Val Loss : {val_loss}")
                    pbar.update(1)
                self.logger.info(f"Epoch {self.epoch} ended with train loss {total_loss / epoch_steps} validation loss {val_loss}")


    def predict(self, batch):
        raise NotImplementedError

    def save_checkpoint(self):
        save_path = f"{self.checkpoint_dir}/checkpoint_epoch-{self.epoch}-step-{self.global_step}.pth"
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }, save_path)

"""


class Trainer:
    def __init__(
            self,
            config: ModelConfig,
            checkpoint_dir: str,
            log_file: str,
            wandb_project_name: str,
            wandb_entity: str,
            wandb_run_name: str,
            model: nn.Module,
            loss_fn,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
            compile: bool = True,
            device: str | None = "cuda",
            use_fabric: bool = True,  # NEW
    ) -> None:

        self.use_fabric = use_fabric

        if self.use_fabric:
            self.fabric = Fabric(accelerator=device or "auto", precision="16-mixed")
            self.fabric.launch()
            self.device = self.fabric.device
        else:
            self.fabric = None
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compile = compile
        self.checkpoint_dir = checkpoint_dir
        self.config = config
        self.epoch: int = 0
        self.global_step: int = 0

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
        formatter = logging.Formatter("{levelname:<8} {message}", style="{")
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        if self.compile:
            self.model = torch.compile(self.model)
            self.logger.info("Model is compiled!")

        if self.use_fabric:
            self.model, self.optimizer = self.fabric.setup(
                self.model, self.optimizer
            )
        else:
            self.model = self.model.to(self.device)

        wandb.init(
            entity=wandb_entity,
            project=wandb_project_name,
            name=wandb_run_name,
            config=dataclasses.asdict(config)
        )


    def _common_step(self, batch) -> torch.Tensor:
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        labels = torch.where(
            (labels >= 0) & (labels < self.model.config.vocab_size),
            labels,
            -100
        )

        outputs = self.model(input_ids, attention_mask)
        loss = self.loss_fn(
            outputs.view(-1, outputs.size(-1)), labels.view(-1)
        )
        return loss


    def evaluate(self, val_dataloader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                loss = self._common_step(batch)
                total_loss += loss.item()

        return total_loss / len(val_dataloader)


    def train(
            self,
            train_dataloader,
            val_dataloader=None,
            num_epochs: int = 1,
            eval_every: int = 5000,
            save_every: int = 5000,
            grad_accumulation_steps: int = 1,
            grad_clip_max_norm: float = 1.0
    ) -> None:

        if self.use_fabric:
            train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
            if val_dataloader:
                val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            total_loss = 0.0
            tokens_seen = 0
            epoch_steps = 0
            val_loss = None

            with tqdm(total=len(train_dataloader),
                      desc=f"Training Epoch {self.epoch}",
                      dynamic_ncols=True) as pbar:

                for idx, batch in enumerate(train_dataloader):
                    tokens_seen += batch['input_ids'].numel()
                    self.model.train()

                    loss = self._common_step(batch)
                    loss_scaled = loss / grad_accumulation_steps

                    if self.use_fabric:
                        self.fabric.backward(loss_scaled)
                    else:
                        loss_scaled.backward()

                    if ((idx + 1) % grad_accumulation_steps == 0) or (idx + 1 == len(train_dataloader)):

                        if self.use_fabric:
                            self.fabric.clip_gradients(
                                self.model,
                                self.optimizer,
                                max_norm=grad_clip_max_norm
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=grad_clip_max_norm
                            )

                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                        if self.scheduler:
                            self.scheduler.step()

                    self.global_step += 1
                    epoch_steps += 1
                    total_loss += loss_scaled.item()

                    self.logger.info(
                        f"Train epoch: {self.epoch} step: {self.global_step} "
                        f"Tokens seen: {tokens_seen} Train Loss: {loss.item()}"
                    )

                    current_lr = self.optimizer.param_groups[0]['lr']
                    wandb.log(
                        {
                            'train_loss': loss.item(),
                            'tokens_seen': tokens_seen,
                            'epoch': self.epoch,
                            'lr': current_lr,
                            'step': self.global_step
                        },
                        step=self.global_step
                    )

                    if (
                        self.global_step % eval_every == 0
                        and val_dataloader
                        and self.global_step > 0
                    ) or epoch_steps == len(train_dataloader):

                        val_loss = self.evaluate(val_dataloader)
                        self.logger.info(
                            f"Eval Step: {self.global_step} "
                            f"Tokens seen: {tokens_seen} Val Loss : {val_loss}"
                        )

                        wandb.log(
                            {
                                'val_loss': val_loss,
                                'tokens_seen': tokens_seen,
                                'epoch': self.epoch,
                                'step': self.global_step
                            },
                            step=self.global_step
                        )

                    if self.global_step % save_every == 0:
                        self.save_checkpoint()
                        self.logger.info(
                            f"Checkpoint Saved | Train loss: {loss} | Eval loss: {val_loss}"
                        )

                    pbar.set_description(
                        f"Train Loss step : {loss.item():.4f} | Val Loss : {val_loss}"
                    )
                    pbar.update(1)

                self.logger.info(
                    f"Epoch {self.epoch} ended with train loss "
                    f"{total_loss / epoch_steps} validation loss {val_loss}"
                )


    def predict(self, batch):
        raise NotImplementedError


    def save_checkpoint(self):
        save_path = f"{self.checkpoint_dir}/checkpoint_epoch-{self.epoch}-step-{self.global_step}.pth"

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }

        if self.use_fabric:
            self.fabric.save(save_path, state)
        else:
            torch.save(state, save_path)
