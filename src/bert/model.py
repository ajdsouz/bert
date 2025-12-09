from dataclasses import dataclass

import torch.nn as nn
from torchtyping import TensorType as T
from typeguard import typechecked

from .layers import EmbeddingLayer, SinusoidalPositionalEncoding, EncoderLayer


@dataclass
class BERTConfigTemplate:
    block_size: int
    d_model: int
    d_ffn: int
    n_heads: int
    n_layer: int
    dropout: float
    vocab_size: int


class BERTTestConfig(BERTConfigTemplate):
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
    vocab_size = 30522


class BertEncoder(nn.Module):
    def __init__(self, config: BERTConfigTemplate):
        super().__init__()
        self.config = config
        # fmt: off
        self.transformer = nn.ModuleDict(dict(
            spe = SinusoidalPositionalEncoding(d_model=config.d_model, block_size=config.block_size),
            wte = EmbeddingLayer(vocab_size=config.vocab_size, d_model=config.d_model),
            h = nn.ModuleList([
                EncoderLayer(d_model=config.d_model, n_heads=config.n_heads, d_ffn=config.d_ffn)
                for _ in range(config.n_layer)
            ]),
            ln_f = nn.LayerNorm(config.d_model),
        ))
        # fmt: on
        self.head = nn.Linear(
            in_features=config.d_model, out_features=config.vocab_size
        )

    @typechecked
    def forward(
        self,
        input_ids: T["batch", "sequence", int],
        attention_mask: T["batch", "sequence", int] | None = None,
    ):
        """Bert model implementation

        Args:
            input_ids: `T["batch", "sequence", int]` Indices in the vocabulary.
            attention_mask: `T["batch", "sequence", int]`
                Attention mask corresponding to `input_ids`. Defaults to `None`.
        Returns:
            `T["batch", "sequence" "vocab"]` Model output.
        """
        B, S = input_ids.shape
        D = self.config.d_model

        tok_emb: T[B * S, D] = self.transformer.wte(input_ids.reshape(-1))
        x: T[B, S, D] = self.transformer.spe(tok_emb.reshape(B, S, D))
        for block in self.transformer.h:
            x: T[B, S, D] = block(x, attention_mask)
        x = self.transformer.ln_f(x)
        logits: T["batch", "sequence", "vocab"] = self.head(x)
        return logits
