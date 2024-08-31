import torch
import torch.nn as nn
from utilities import utils

class MatrixFactorization(utils.NNBase):
    def __init__(self, n_users, n_items, n_factors, device='cpu', dtype=torch.float32):
        super(MatrixFactorization, self).__init__()
        self.save_hyperparamters()
        
        self.P = nn.Embedding(n_users, n_factors, device=device, dtype=dtype)
        self.Q = nn.Embedding(n_items, n_factors, device=device, dtype=dtype)
        self.b_user = nn.Embedding(n_users, 1, device=device, dtype=dtype)
        self.b_item = nn.Embedding(n_items, 1, device=device, dtype=dtype)
        
@utils.add_to_class(MatrixFactorization)
def forward(self, user_id, item_id):
    p_u = self.P(user_id)
    q_i = self.Q(item_id)
    b_u = self.b_user(user_id)
    b_i = self.b_item(item_id)
    
    return torch.dot(p_u, q_i) + b_u + b_i

        
