import math

import torch
from torchtyping import TensorType as T
from typeguard import typechecked


@typechecked
def attention(
    query: T["batch", "weight"],
    key: T["batch", "weight"],
    value: T["batch", "weight"],
    scale: float | None = None,
    mask: T["batch", int] | None = None,
) -> tuple[T["batch", "weight"], T["batch", "batch"]]:
    """Computes attention score as in https://arxiv.org/abs/1706.03762

    Args:
        query: `T["batch", "weight"]` Query weight tensor.
        key: `T["batch", "weight"]` Key weight tensor.
        value: `T["batch", "weight"]` Value weight tensor.
        scale: Scaling factor for attention calculation.
            If no scale is provided, will use last dim of key. Defaults to None.
        mask: `T["batch", int]` Mask for masking tokens. Defaults to None.

    Returns:
        A tuple (attention_score, attention_weights), where
        - attention_score: `T["batch", "weight"]`.
        - attention_weights: `T["batch", "batch"]`.
    """
    if scale is None:
        scale: float = math.sqrt(key.size(-1))

    scores: T["batch", "batch"] = (query @ key.transpose(-2, -1)) / scale

    if mask is not None:
        scores = scores.masked_fill(mask == 0, value=float("-inf"))

    attention_weights: T["batch", "batch"] = torch.softmax(scores, -1)
    attention: T["batch", "weight"] = attention_weights @ value

    return attention, attention_weights


@typechecked
def split_heads(
    x: T["batch", "sequence", "d_model"],
    n_heads: int,
) -> T["batch", "n_heads", "sequence", "d_head"]:
    """Split tensor into n_heads tensors for individual attention heads

    Args:
        x: `T["batch", "sequence", "d_model"]` tensor to be split.
        n_heads: Number of heads.

    Returns:
        `T["batch", "n_heads", "sequence", "d_head"]` A different view of the tensor.
    """
    B, S, D = x.shape
    d_head = D // n_heads
    x: T[B, n_heads, S, D] = x.view(B, S, n_heads, d_head).transpose(1, 2)
    return x


@typechecked
def merge_heads(
    x: T["batch", "n_heads", "sequence", "d_head"],
) -> T["batch", "sequence", "d_model"]:
    """Merge tensor after after attention calculation

    Args:
        x: `T["batch", "n_heads", "sequence", "d_head"]` Tensor to be merged.

    Returns:
        `T["batch", "sequence", "d_model"]` Merged tensor.
    """
    B, Nh, S, Dh = x.shape
    x: T[B, S, Nh, Dh] = x.transpose(1, 2).contiguous()
    return x.view(B, S, Nh * Dh)
