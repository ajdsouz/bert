from bert.functional import split_heads, merge_heads

import torch

tensor = torch.randn((16, 128, 256))

split = split_heads(tensor, 8)
print(split.shape)

merge = merge_heads(split)
print(merge.shape)