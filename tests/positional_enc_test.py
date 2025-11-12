# ------ Testing positional encoding
from src.bert.layers import SinusoidalPositionalEncoding

import torch
import torch.nn as nn

torch.manual_seed(1337)


tensor = torch.rand((4, 4))

spe = SinusoidalPositionalEncoding(4, 4)
print("tensor :", tensor)
print("----------------------------------")
print("sample :",spe(tensor))
print("----------------------------------")
print("sample shape :",spe(tensor).shape)

print("----------------------------------")
batched_tensor = torch.rand((2,4,4))
print("tensor :", batched_tensor)
print("----------------------------------")
print("Batched tensor :",spe(batched_tensor))
print("----------------------------------")
print("Batched tensor shape :", spe(batched_tensor).shape)