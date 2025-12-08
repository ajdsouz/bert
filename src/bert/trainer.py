import wandb
import logging
from tqdm import tqdm

import torch
import torch.nn as nn

class Trainer:
    def __init__(
            self,
            checkpoint_dir: str,
            log_file: str,
            model: nn.Module,
            loss_fn: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
            precision: str = 'bf16',
            tracking: bool = False,
            wandb_project_name: str | None = None,
            wandb_entity: str | None = None,
            compile: bool = True,
            device: str | None = "cuda"
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
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compile = compile
        self.checkpoint_dir = checkpoint_dir
        self.tracking = tracking
        self.epoch: int = 0
        self.global_step: int = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # TODO add mode precisions (fp8/fp4/etc)
        precision_dict = {
            'mixed-16':torch.float16,
            'bf16':torch.bfloat16,
            'fp32':torch.float32
        }
        self.precision = precision_dict[precision]

        
        handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
        formatter = logging.Formatter("{levelname:<8} {message}", style="{")
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        self.logger.info(f"Using precision = {self.precision}")
        if self.compile:
            torch.compile(self.model)
            self.logger.info("Model is compiled!")
        else:
            self.logger.info("Model is not compiled")

        parameters = self.compute_params()
        self.logger.info(f"Model has {parameters} parameters")

        if self.tracking:
            wandb.init(
                entity=wandb_entity,
                project=wandb_project_name,
            )
    
    def compute_params(self) -> int:
        """Computes trainable parameters in the model provided

        Returns:
            int: trainable parameter count
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _common_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes masked language modelling forward pass

        Args:
            batch (dict[str, torch.Tensor]): batch of data

        Returns:
            torch.Tensor: loss tensor
        """
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
        labels =batch['labels'].to(self.device, non_blocking=True)
        labels = torch.where((labels >= 0) & (labels < self.model.config.vocab_size), labels, -100)
        outputs = self.model(input_ids, attention_mask)
        loss = self.loss_fn(
            outputs.view(-1, outputs.size(-1)), labels.view(-1)
        )
        return loss
    

    def evaluate(self, val_dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluates model on validation dataset

        Args:
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader

        Returns:
            float: total loss for all batches in validation dataset
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                if self.precision != torch.float32:
                    with torch.amp.autocast(device_type=self.device, dtype=self.precision):
                        loss = self._common_step(batch)
                else:
                    loss = self._common_step(batch)
                total_loss += loss.item()

        return total_loss / len(val_dataloader)


    def train(
            self, 
            train_dataloader: torch.utils.data.DataLoader, 
            val_dataloader: torch.utils.data.DataLoader | None,
            num_epochs: int | None = 1,
            eval_every: int | None = 5000,
            save_every: int | None = 5000,
            grad_accumulation_steps: int = 1,
            grad_clip_max_norm: float = 1.0
    ) -> None:
        """Training loop of the trainer

        Args:
            train_dataloader (torch.utils.data.DataLoader): Training dataloader
            val_dataloader (torch.utils.data.DataLoader | None): Validation dataloader
            num_epochs (int, optional): number of epochs to train the model for. Defaults to 1.
            eval_every (int, optional): Evaluate model after every ```eval_every``` no. of steps. Defaults to 5000.
            save_every (int, optional): Save checkpoint after every ```save_every``` steps. Defaults to 5000.
            grad_accumulation_steps (int, optional): Accumulate gradients over ```grad_accumulation_steps```. Defaults to 1, so no grad accumulation.
            grad_clip_max_norm (float, optional): clip gradients. Defaults to 1.0.
        """
        
        if self.precision != torch.float32:
            scaler = torch.amp.GradScaler()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            total_loss = 0.0
            tokens_seen = 0
            epoch_steps = 0
            val_loss = 0.0
            pbar = tqdm(
                total=len(train_dataloader),
                bar_format="{l_bar}{bar:40}| {n_fmt}/{total_fmt} {percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                colour="magenta",           
                dynamic_ncols=True,
                )
            for idx, batch in enumerate(train_dataloader):

                # --------- Training --------
                # tokens_seen += (batch['input_ids'].size(0) * batch['input_ids'].size(1))
                tokens_seen += batch['input_ids'].numel() # store no. of tokens model has seen in each batch
                self.model.train()

                if self.precision != torch.float32:
                    with torch.amp.autocast(device_type=self.device, dtype=self.precision):
                        loss = self._common_step(batch)
                        loss_scaled = loss / grad_accumulation_steps
                    
                    scaler.scale(loss_scaled).backward()

                    if ((idx + 1) % grad_accumulation_steps == 0) or (idx + 1 == len(train_dataloader)):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=grad_clip_max_norm
                        )
                            
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.scheduler:
                            self.scheduler.step()
                else:
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
                total_loss += loss.item()

                pbar.update(1)
                pbar.set_postfix(
                    epoch=f"{self.epoch}",
                    steps=f"{self.global_step}",
                    train_loss=f"{total_loss:.4}",
                    tokens=f"{tokens_seen}",
                    val_loss=f"{val_loss}"
                )


                self.logger.info(f"Train epoch: {self.epoch} step: {self.global_step} Tokens seen: {tokens_seen} Train Loss: {loss.item()}")
                wandb.log({'train_loss': loss.item(), 'tokens_seen': tokens_seen, 'epoch': self.epoch, 'step': self.global_step}, step=self.global_step) if self.tracking else None
            
                # --------- evaluation ---------
                if self.global_step % eval_every == 0 and val_dataloader and self.global_step > 0:
                    val_loss = self.evaluate(val_dataloader)
                    self.logger.info(f"Eval Step: {self.global_step} Tokens seen: {tokens_seen} Val Loss : {val_loss}")
                    wandb.log({'val_loss': val_loss, 'tokens_seen': tokens_seen, 'epoch': self.epoch, 'step': self.global_step}, step=self.global_step) if self.tracking else None

                # --------- checkpointing ---------
                if self.global_step % save_every == 0:
                    self.save_checkpoint()
                    self.logger.info(f"Checkpoint Saved | Train loss: {loss} | Eval loss: {val_loss}")

            pbar.close()

            self.logger.info(f"Epoch {self.epoch} ended with train loss {total_loss / epoch_steps} validation loss {val_loss}")


    def predict(self, batch: dict[str, torch.Tensor]):
        raise NotImplementedError
    
    def load_model(self):
        raise NotImplementedError

    def save_checkpoint(self) -> None:
        """Checkpoint Saving
        """
        save_path = f"{self.checkpoint_dir}/checkpoint_epoch{self.epoch}_step{self.global_step}.pth"
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }, save_path)