import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F 

from .functional import attention, split_heads, merge_heads
from .config import ModelConfig

ACT2FN = {
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
    'gelu': nn.GELU,
}



class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
NORM2FN = {
    'layernorm': nn.LayerNorm,
    'rmsnorm': RMSNorm,
}

class FFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(in_features=config.d_model, out_features=config.d_ffn, bias=True)
        self.fc2: nn.Linear = nn.Linear(in_features=config.d_ffn, out_features=config.d_model, bias=True)
        self.dropout: nn.Dropout = nn.Dropout(p=config.hidden_dropout)
    def forward(self, x: Tensor) -> Tensor:
        """Feed-forward layer with ReLU activation

        Args:
            x (Tensor): Input Tensor

        Returns:
            Tensor: Output Tensor
        """
        return self.fc2(F.relu(self.fc1(x)))

class GatedMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up_proj: nn.Linear = nn.Linear(in_features=config.d_model, out_features=config.d_ffn, bias=True)
        self.gate_proj: nn.Linear = nn.Linear(in_features=config.d_model, out_features=config.d_ffn, bias=True)
        self.down_proj: nn.Linear = nn.Linear(in_features=config.d_ffn, out_features=config.d_model, bias=True)
        self.activation: nn.Module = ACT2FN[config.activation]()

    def forward(self, x: torch.Tensor):
        x, swish = self.up_proj(x), self.activation(self.gate_proj(x))
        down_proj = self.down_proj(x * swish)
        return down_proj


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO : add functionality to log attention scores for interp
        self.n_heads = config.n_heads
        self.d_heads = config.d_model // config.n_heads
        self.dropout = config.attention_dropout
        # self.log_attention : bool = log_attention
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj: nn.Linear = nn.Linear(config.d_model, config.d_model)
        self.Wq : nn.Linear = nn.Linear(config.d_model, config.d_model)
        self.Wk : nn.Linear = nn.Linear(config.d_model, config.d_model)
        self.Wv : nn.Linear= nn.Linear(config.d_model, config.d_model)
    
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Multi-head attention

        Args:
            x (Tensor): Input Tensor

        Returns:
            Tensor: OutPut Tensor
        """
        q = split_heads(self.Wq(x), self.n_heads)
        k = split_heads(self.Wk(x), self.n_heads)
        v = split_heads(self.Wv(x), self.n_heads)

        if mask is not None:
            mask = mask[:, None, None, :] # autobroadcast to [B, H, S, S]

        att, weights = attention(q, k, v, mask=mask, p=self.dropout)

        attention_out = merge_heads(att)

        return self.out_proj(attention_out)
    

class EmbeddingLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.embedding_table: nn.Embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embedding_table.weight, mean=0.0, std=0.02)

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
    def __init__(self, config: ModelConfig):
        super().__init__()
        position: Tensor = torch.arange(config.block_size).unsqueeze(1)
        div_terms: Tensor = torch.exp(
            torch.arange(0, config.d_model, 2) * (-torch.log(torch.tensor(10000.0)) / config.d_model)
        )
        pe: Tensor = torch.zeros(config.block_size, config.d_model)
        pe[:,0::2] = torch.sin(position * div_terms)
        pe[:,1::2] = torch.cos(position * div_terms)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)


    def forward(self, x: Tensor) -> Tensor:
        """Generate Sinosoidal Positional Embeddings

        Args:
            x (Tensor): Word Embedding

        Returns:
            Tensor: Word embedding + positional information
        """
        return x + self.pe[:, :x.size(1), :]

MLP2FN = {
    'vanilla' : FFN,
    'gated' : GatedMLP,
}



class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # DONE : added MLP2FN to experiment with multiple mlp types. rn supports vanilla mlp and gated mlp
        # TODO : add flexible NORM2FN to experiment with multiple types.
        # Done : add norm_position feature to switch between pre and post norm.

        self.config = config
        self.ln1  = NORM2FN[config.norm_type](config.d_model, eps=config.layernorm_eps)
        self.ln2  = NORM2FN[config.norm_type](config.d_model, eps=config.layernorm_eps)
        self.mha: MultiHeadAttention = MultiHeadAttention(config)
        self.mlp: FFN = MLP2FN[config.mlp_type](config)
    
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Encoder layer / block for encoder models

        Args:
            x (Tensor): Tensor Input

        Returns:
            Tensor: Tensor Output
        """
        hidden_state = x
        if self.config.norm_position == 'postnorm':
            hidden_state = self.ln1(
                hidden_state + self.mha(x, mask)
            )
            return self.ln2(
                hidden_state + self.mlp(hidden_state)
            )
        else:
            hidden_state = hidden_state + self.mha(self.ln1(x), mask)
            return hidden_state + self.mlp(self.ln2(hidden_state))



