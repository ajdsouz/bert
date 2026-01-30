import torch


def build_lr_scheduler_linear_warmup_then_cosine(
    optimizer: torch.optim.Optimizer,
    linear_start_factor: float,
    linear_end_factor: float,
    linear_steps: int,
    cosine_min_lr: float,
    total_steps: int,
) -> torch.optim.lr_scheduler.SequentialLR:
    linear = torch.optim.lr_scheduler.LinearLR(
        optimizer, linear_start_factor, linear_end_factor, linear_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_steps - linear_steps, eta_min=cosine_min_lr
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [linear, cosine], [linear_steps]
    )
    return scheduler
