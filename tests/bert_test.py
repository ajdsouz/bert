from bert.model import BertEncoder, BERTTestConfig

import torch
import torch.nn as nn

tensor = torch.randint(low=10, high=100, size=(1, 64))

config = BERTTestConfig
model = BertEncoder(config = config)

with torch.no_grad():
    out = model(tensor)
print("model output : ", out)
print("----------------------------------")
print("output shape :", out.shape)

