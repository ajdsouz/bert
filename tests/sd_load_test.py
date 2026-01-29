import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from bert.model import BertEncoder, BERTConfigTemplate

path = 'ckpt/last.ckpt'

class STSBConfig(BERTConfigTemplate):
    block_size = 256
    d_model = 512
    d_ffn = 1024
    n_heads = 8
    n_layer = 6
    dropout = 0.0
    vocab_size = 50368

model = BertEncoder(STSBConfig)
ckpt = torch.load(path, map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=False)
print("loaded model")