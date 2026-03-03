from .layers import EmbeddingLayer, SinusoidalPositionalEncoding, EncoderLayer
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


class BertEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
                spe = SinusoidalPositionalEncoding(config=config),
                wte = EmbeddingLayer(config=config),
                h = nn.ModuleList([
            EncoderLayer(config=config) for _ in range(config.n_layer)
                ]),
                ln_f = nn.LayerNorm(config.d_model)
            ))
        self.head = nn.Linear(in_features=config.d_model, out_features=config.vocab_size)        

    def forward(self, input_ids: Tensor, attention_mask: Tensor):
        """Bert model implementation

        Args:
            input_ids (Tensor): tokenizer output [Batch Sequence]

        Returns:
            _type_: model output [Batch Sequence Vocab]
        """
        tok_emb = self.transformer.wte(input_ids)
        x = self.transformer.spe(tok_emb)
        for block in self.transformer.h:
            x = block(x, attention_mask)
        x = self.transformer.ln_f(x)
        logits = self.head(x)
        return logits
