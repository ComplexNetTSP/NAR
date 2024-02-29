import torch
import torch.nn as nn

class NodeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
      super().__init__()
      self.hidden_dim = hidden_dim
      self.input_dim = input_dim
      self.lin = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
      if x.dim() == 1:
        x = x.unsqueeze(-1)
      x = self.lin(x)
      return x
      
class Encoder(nn.Module):
  def __init__(self, hidden_dim=128):
    super(Encoder, self).__init__()
    self.hidden_dim = hidden_dim
    self.encoder = NodeEncoder(1, self.hidden_dim)
    
  def forward(self, Xfeatures):
    return self.encoder(Xfeatures)