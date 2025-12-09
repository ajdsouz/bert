import torch
from torch.nn.functional import scaled_dot_product_attention
from torchtyping import patch_typeguard

from bert.functional import attention


patch_typeguard()


x = torch.randn((2, 3))

sdpa_val = scaled_dot_product_attention(x, x, x)
my_att = attention(x, x, x)
print("my attention : ", my_att[0])
print("sdpa : ", sdpa_val)
assert torch.allclose(my_att[0], sdpa_val)


mask = torch.tensor([1, 0])
sdpa_val = scaled_dot_product_attention(x, x, x, attn_mask=mask.bool())
my_att = attention(x, x, x, mask=mask)
print("my masked attention : ", my_att[0])
print("sdpa masked : ", sdpa_val)
assert torch.allclose(my_att[0], sdpa_val)
