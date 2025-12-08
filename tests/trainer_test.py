from bert.model import BertEncoder, BERTBaseConfig
from bert.trainer import Trainer
import torch
import torch.nn as nn

class BertTestConfig(BERTBaseConfig):
    block_size = 256
    d_model = 256
    d_ffn = 1024
    n_heads = 6
    n_layer = 6
    dropout = 0.0


model = BertEncoder(config = BertTestConfig)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

trainer = Trainer(
    checkpoint_dir='ckpt',
    log_file='logs/logfile.log',
    tracking=False,
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device="cpu",
)

""" 
class BertTestConfig(BERTBaseConfig):
    block_size = 128
    d_model = 256
    d_ffn = 512
    n_heads = 6
    n_layer = 6
    dropout = 0.0

>> param count : 18820922

class BertTestConfig(BERTBaseConfig):
    block_size = 128
    d_model = 256
    d_ffn = 1024
    n_heads = 6
    n_layer = 6
    dropout = 0.0

>> param count : 20396858
"""