import torch
from torchtyping import patch_typeguard

from bert.functional import split_heads, merge_heads


patch_typeguard()


tensor = torch.randn((16, 128, 256))

split = split_heads(tensor, 8)
print(split.shape)
assert (*split.shape,) == (16, 8, 128, 256//8)

merge = merge_heads(split)
print(merge.shape)
assert (*merge.shape,) == (16, 128, 256)
