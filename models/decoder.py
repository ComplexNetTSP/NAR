import torch.nn as nn 

class BaseEdgeDecoder(nn.Module):
  def __init__(self, hidden_dim):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.source_lin = nn.Linear(hidden_dim, hidden_dim)
    self.target_lin = nn.Linear(hidden_dim, hidden_dim)
  
  def forward(self, hiddens, edge_index):
    zs = self.source_lin(hiddens) # N x H
    zt = self.target_lin(hiddens) # N x H
    return (zs[edge_index[0]] * zt[edge_index[1]]).sum(dim=-1) # M x 1
      
class EdgeMaskDecoder(BaseEdgeDecoder):
  def __init__(self, hidden_dim):
    super().__init__(hidden_dim)

  def forward(self, hiddens, edge_index):
    out = super().forward(hiddens, edge_index).sigmoid().squeeze(-1)
    return out