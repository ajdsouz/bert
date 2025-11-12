import math
from torch import Tensor 
import torch.nn.functional as F 


def attention(
        query: Tensor, 
        key: Tensor, 
        value: Tensor, 
        scale: float | None = None, 
        mask : Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Computes attention score as in https://arxiv.org/abs/1706.03762

    Args:
        query (Tensor): query weight tensor
        key (Tensor): key weight tensot
        value (Tensor): value weight tensor
        scale (float, optional): scaling factor for attention calculation. 
            If no scale is provided, will use last dim of key. Defaults to None.
        mask (Tensor, optional): mask for masking tokens. Defaults to None.

    Returns:
        tuple [Tensor, Tensor]: attention score, attention weights
    """
    if scale is None:
        scale = math.sqrt(key.size(-1))

    scores = (query @ key.transpose(-2, -1)) / scale

    if mask is not None:
        scores.masked_fill(mask==0, value=float('-inf'))
    
    attention_weights = F.softmax(scores, -1)
    attention = attention_weights @ value 

    return attention, attention_weights


def split_heads(x: Tensor, n_heads: int) -> Tensor:
    """Split tensor into n_heads tensors for individual attention heads

    Args:
        x (Tensor): tensor to be split
        n_heads (int): number of heads

    Returns:
        Tensor: A different view of the tensor of form [Batch n_heads Sequence d_head]
    """
    # [batch sequence d_model]
    B, S, D = x.shape
    d_head = D // n_heads
    # [batch sequence n_heads d_head] => [batch n_heads sequence d_head]
    return x.view(B, S, n_heads, d_head).transpose(1, 2)

def merge_heads(x: Tensor) -> Tensor:
    """Merge tensor after after attention calculation

    Args:
        x (Tensor): Tensor [Batch n_heads Sequence d_head] to be merged

    Returns:
        Tensor: Merged tensor of shape [Batch Sequence d_model]
    """
    B, Nh, S, Dh = x.shape
    return x.transpose(1, 2).contiguous().view(B, S, Nh * Dh)
 
 
 
 
 
 
 
 
 
 
 
