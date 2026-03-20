from bert.model import BertEncoder, BertForMLM, BertModel
from bert.config import ModelConfig
import torch
import torch.nn as nn

tensor = torch.randint(low=10, high=100, size=(1, 64))

config = ModelConfig(block_size=128, d_model=256, d_ffn=512, hidden_dropout=0.0, attention_dropout=0.1, n_heads=4, n_layer=2, vocab_size=50265, activation='relu', mlp_type='vanilla', norm_position='postnorm', layernorm_eps=1e-12, norm_type='rmsnorm')
model1 = BertEncoder(config = config)
model2 = BertModel(config)
model3 = BertForMLM(config)

print(model1)
print("-----------------")

print(model2)

print("-----------------")

print(model3)
# with torch.no_grad():
#     out = model(tensor)
# print("model output : ", out)
# print("----------------------------------")
# print("output shape :", out.shape)

