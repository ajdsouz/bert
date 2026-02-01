from bert.model import ModelConfig

import dataclasses

bertconfig = ModelConfig(
    block_size=128,
    d_model=8,
    d_ffn=32,
    n_heads=2,
    n_layer=2,
    dropout=0.0,
    vocab_size=2,
)

print(dataclasses.asdict(bertconfig))
