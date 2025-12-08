import torch
import torch.nn as nn
import torch.nn.functional as F 

#----- parity check between my attention and pytorch's sdpa 
from bert.functional import attention
from torch.nn.functional import scaled_dot_product_attention
#-----------------------------------------------------------


x = torch.randn((2, 3))

sdpa_val = scaled_dot_product_attention(x, x, x)

my_att = attention(x, x, x)
print("my attention : ", my_att)
print("sdpa : ", sdpa_val)
print(torch.allclose(my_att, sdpa_val)) # should return true





