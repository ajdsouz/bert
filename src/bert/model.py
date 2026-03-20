from .layers import EmbeddingLayer, SinusoidalPositionalEncoding, EncoderLayer, NORM2FN
from .config import ModelConfig
from dataclasses import dataclass
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

"""@dataclass
class ModelConfig:
    block_size: int  
    d_model: int
    d_ffn: int
    n_heads: int
    n_layer: int
    dropout: float
    vocab_size: int"""

"""class BERTTestConfig(BERTConfigTemplate):
    block_size: int = 64
    d_model: int = 64
    d_ffn: int = 256
    n_heads: int = 2
    n_layer: int = 2
    dropout: float = 0.0
    vocab_size: int = 500

class BERTBaseConfig(BERTConfigTemplate):
    d_model = 768
    d_ffn = 3072
    n_heads = 12
    n_layer = 12
    dropout = 0.0
    vocab_size = 30522"""

class BertModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wte = EmbeddingLayer(config)
        self.wpe = SinusoidalPositionalEncoding(config)
        if config.norm_position == "postnorm":
            self.ln_e = NORM2FN[config.norm_type](config.d_model, eps=config.layernorm_eps)
        else:
            self.ln_f = NORM2FN[config.norm_type](config.d_model, eps=config.layernorm_eps)
        self.dropout_e = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            EncoderLayer(config) for _ in range(config.n_layer)
        )
        

    def forward(self, input_ids: Tensor, attention_mask: Tensor):
        x = self.wpe(self.wte(input_ids))

        if self.config.norm_position == 'postnorm':
            x = self.ln_e(x)

        x = self.dropout_e(x)
        for layer in self.layers:
            x = layer(x, attention_mask)

        if self.config.norm_position == 'prenorm':
            x = self.ln_f(x)
        
        return x
    
class BertForMLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = BertModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.lm_head.weight = self.model.wte.embedding_table.weight

    def forward(self, input_ids, attentin_mask):
        x = self.model(input_ids, attentin_mask)
        logits = self.lm_head(x)
        return logits


