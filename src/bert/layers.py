import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F 

from .functional import attention, split_heads, merge_heads

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, bias: bool = False):
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(in_features=d_model, out_features=d_ffn, bias=True)
        self.fc2: nn.Linear = nn.Linear(in_features=d_ffn, out_features=d_model, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Feed-forward layer with ReLU activation

        Args:
            x (Tensor): Input Tensor

        Returns:
            Tensor: Output Tensor
        """
        return self.fc2(F.relu(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model : int, n_heads : int, log_attention : bool =False):
        super().__init__()
        # TODO : add functionality to log attention scores for interp

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_heads = self.d_model // self.n_heads
        self.log_attention : bool = log_attention
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj: nn.Linear = nn.Linear(self.d_model, self.d_model)
        self.Wq : nn.Linear = nn.Linear(self.d_model, self.d_model)
        self.Wk : nn.Linear = nn.Linear(self.d_model, self.d_model)
        self.Wv : nn.Linear= nn.Linear(self.d_model, self.d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        """Multi-head attention

        Args:
            x (Tensor): Input Tensor

        Returns:
            Tensor: OutPut Tensor
        """
        q = split_heads(self.Wq(x), self.n_heads)
        k = split_heads(self.Wk(x), self.n_heads)
        v = split_heads(self.Wv(x), self.n_heads)

        att, weights = attention(q, k, v)

        attention_out = merge_heads(att)

        return self.out_proj(attention_out)
    

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding_table: nn.Embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Embedding layer for word embeddings. Does not have positional information.
        Add Positional embeddings yourself.

        Args:
            token_ids (Tensor): Tokenized input

        Returns:
            Tensor: Word embeddings
        """
        embeddings = self.embedding_table(token_ids)
        return math.sqrt(self.d_model) * embeddings

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, block_size: int):
        super().__init__()
        position: Tensor = torch.arange(block_size).unsqueeze(1)
        div_terms: Tensor = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe: Tensor = torch.zeros(block_size, d_model)
        pe[:,0::2] = torch.sin(position * div_terms)
        pe[:,1::2] = torch.cos(position * div_terms)

        self.register_buffer("pe", pe.unsqueeze(0))


    def forward(self, x: Tensor) -> Tensor:
        """Generate Sinosoidal Positional Embeddings

        Args:
            x (Tensor): Word Embedding

        Returns:
            Tensor: Word embedding + positional information
        """
        return x + self.pe[:, :x.size(1), :]


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_heads = n_heads
        self.ln1: nn.LayerNorm = nn.LayerNorm(self.d_model)
        self.ln2: nn.LayerNorm = nn.LayerNorm(self.d_model)
        self.mha: MultiHeadAttention = MultiHeadAttention(self.d_model, self.n_heads)
        self.ffn: FFN = FFN(self.d_model, self.d_ffn)
    
    def forward(self, x: Tensor) -> Tensor:
        """Encoder layer / block for encoder models

        Args:
            x (Tensor): Tensor Input

        Returns:
            Tensor: Tensor Output
        """
        x = self.ln1(
            x + self.mha(x)
        )
        return self.ln2(
            x + self.ffn(x)
        )



