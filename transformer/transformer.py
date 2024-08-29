import torch
import torch.nn as nn
from utils import NNBase, add_to_class, xavier_init

def PositionalEncoding(NNBase):
    def __init__(self, d_model, max_len=5000, device='cpu', dropout=0.1, dtype=torch.float32):
        super(PositionalEncoding, self).__init__()
        self.save_hyperparameters()
        
        self.dropout = nn.Dropout(dropout)
        self.pos_encodings = torch.zeros((max_len, d_model), device=device, dtype=dtype, requires_grad=False)
        
        rows = torch.arange(max_len, dtype=dtype, device=device, requires_grad=False).reshape(-1, 1)
        cols = torch.arange(d_model, step=2, dtype=dtype, device=device, requires_grad=False)
        
        args = rows / (10000 ** (2 * cols / d_model))
        
        self.pos_encodings[:, ::2] = torch.sin(args)
        self.pos_encodings[:, 1::2] = torch.cos(args)
        
@add_to_class(PositionalEncoding)
def forward(self, x):
    batch_size, seq_len, d_token = x.shape
    x = x + self.pos_encodings[:seq_len, ]
    return self.dropout(x)

class MultiHeadAttention(NNBase):
    def __init__(self, d_model, h=8, d_k=None, d_v=None, device='cpu', dropout=0.1, dtype=torch.float32):
        super(MultiHeadAttention, self).__init__()
        self.save_hyperparameters()

        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_k if d_k is not None else d_model / h
        self.d_v = d_v if d_v is not None else d_model / h

        # query parameters
        self.w_q = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        xavier_init(self.w_q)
        # key parameters
        self.w_k = nn.Linear(d_model, self.d_k, device=device, dtype=dtype)
        xavier_init(self.w_k)
        # value parameters
        self.w_v = nn.Linear(d_model, self.d_v, device=device, dtype=dtype)
        xavier_init(self.w_v)
        # concat parameters
        self.w_o = nn.Linear(h * self.d_v, d_model, device=device, dtype=dtype)
        # attention block
        self.attention = ScaledDotProductAttention()

@add_to_class(MultiHeadAttention)
def xavier_init(lin_layer):
    torch.init.xavier_uniform_(lin_layer.weight.data)
    torch.init.xavier_uniform_(lin_layer.bias.data)

@add_to_class(MultiHeadAttention)
def forward(self, query, key, value, mask=None):
    
    
        
        



