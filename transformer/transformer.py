import torch
import torch.nn as nn
from utils import NNBase, add_to_class

def PositionalEncoding(NNBase):
    def __init__(self, d_model, max_len=5000, device='cpu', dropout=0.1, dtype=torch.float32):
        super().__init__() 
        self.save_hyperparameters()
        
        self.dropout = nn.Dropout(dropout)
        self.pos_encodings = torch.zeros((max_len, d_model), device=device, dtype=dtype, requires_grad=False)
        
        rows = torch.unsqueeze(torch.arange(max_len, dtype=dtype, device=device, requires_grad=False), dim=1)
        cols = torch.arange(d_model, step=2, dtype=dtype, device=device, requires_grad=False)
        
        args = rows / (10000 ** (2 * cols / d_model))
        
        self.pos_encodings[:, ::2] = torch.sin(args)
        self.pos_encodings[:, 1::2] = torch.cos(args)
        
@add_to_class(PositionalEncoding)
def forward(self, x):
    batch_size, seq_len, d_token = x.shape
    
    x = x + self.pos_encodings[:seq_len, ]
    return self.dropout(x)
    
        
        



