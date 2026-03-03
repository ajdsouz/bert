from dataclasses import dataclass

@dataclass
class ModelConfig:
    block_size: int  
    d_model: int
    d_ffn: int
    n_heads: int
    n_layer: int
    dropout: float
    vocab_size: int
    activation: str
    mlp_type: str
    #norm_type: str # TODO : add RMSNorm as well
    norm_position: str