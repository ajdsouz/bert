from dataclasses import dataclass

@dataclass
class ModelConfig:
    block_size: int  
    d_model: int
    d_ffn: int
    n_heads: int
    n_layer: int
    attention_dropout: float
    hidden_dropout: float
    vocab_size: int
    activation: str
    mlp_type: str
    layernorm_eps: float
    norm_type: str 
    norm_position: str