from bert.model import BertEncoder
from bert.config import ModelConfig
import torch
import torch.nn as nn

tensor = torch.randint(low=10, high=100, size=(1, 64))

config = ModelConfig(block_size=128, d_model=256, d_ffn=512, dropout=0.0, n_heads=4, n_layer=2, vocab_size=50265, activation='silu', mlp_type='gated', norm_position='prenorm')
model = BertEncoder(config = config)
print(model)
# with torch.no_grad():
#     out = model(tensor)
# print("model output : ", out)
# print("----------------------------------")
# print("output shape :", out.shape)

