import torch
from torchtyping import patch_typeguard

from bert.model import BertEncoder, BERTTestConfig


patch_typeguard()


tensor = torch.randint(low=10, high=100, size=(8, 64))

config = BERTTestConfig
model = BertEncoder(config=config)

with torch.no_grad():
    out = model(tensor)
    out = model(tensor, attention_mask=torch.ones((8, 64), dtype=int))
print("model output : ", out)
print("----------------------------------")
print("output shape :", out.shape)
assert (*out.shape,) == (*tensor.shape, config.vocab_size)
assert torch.allclose(out, model(tensor))
