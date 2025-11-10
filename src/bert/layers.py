import math
import torch
import torch.nn as nn
import torch.nn.functional as F 

from src.bert.functional import attention, split_heads, merge_heads

class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, bias = False):
        super().__init__()
        self.fc1 = nn.Linear(in_features=d_model, out_features=d_ffn, bias=True)
        self.fc2 = nn.Linear(in_features=d_ffn, out_features=d_model, bias=True)
        self.activation = F.relu()
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, log_attention=False):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_heads = self.d_model // self.n_heads
        self.log_attention = log_attention
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.Wq = nn.Linear(self.d_heads, self.d_heads)
        self.Wk = nn.Linear(self.d_heads, self.d_heads)
        self.Wv = nn.Linear(self.d_heads, self.d_heads)
    
    def forward(self, x):
        q = split_heads(self.Wq(x), self.n_heads)
        k = split_heads(self.Wk(x), self.n_heads)
        v = split_heads(self.Wv(x), self.n_heads)

        attention_weights = attention(q, k, v)

        attention_out = merge_heads(attention_weights)

        return self.out_proj(attention_out)
    

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding_table = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids):
        embeddings = self.embedding_table(token_ids)
        return math.sqrt(self.d_model) * embeddings

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, block_size):
        super().__init__()
        

    def forward(self):
        pass


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_heads = n_heads
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = FFN(d_model, d_ffn)
    
    def forward(self, x):
        x = F.layer_norm(
            x + self.mha(x),
            self.d_model
        )
        return F.layer_norm(
            x + self.ffn(x),
            self.d_model
        )



