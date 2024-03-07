import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Linear

class MPNN(MessagePassing):
  def __init__(self, in_channels, hidden_channels, activation=None):
    super(MPNN, self).__init__(aggr='max') #  "Max" aggregation.
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.messages = Linear(self.in_channels * 2, self.hidden_channels)
    self.activation = activation

    self.mlp = torch.nn.Sequential(
        Linear(hidden_channels, hidden_channels),
        torch.nn.ReLU(),
        Linear(hidden_channels, self.hidden_channels)
    )
    
  def forward(self, x, edge_index):
    out = self.propagate(edge_index, x=x)
    out = self.mlp(out)
    if self.activation is not None:
      out = self.activation(out)
    return out
    
  def message(self, x_i, x_j):
    # x_i has shape [E, in_channels]
    # x_j has shape [E, in_channels]
    #print('MPNN => xi, xj', x_i.size(), x_j.size())
    tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
    #print('MPNN => messages IN', tmp.size())
    m = self.messages(tmp)
    #print('MPNN => messages OUT', m.size())
    return m
