import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Linear

class MPNN(MessagePassing):
  def __init__(self, in_channels, out_channels):
    super(MPNN, self).__init__(aggr='max') #  "Max" aggregation.
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.messages = Linear(self.in_channels * 2, self.out_channels)
    self.update_fn = Linear(self.in_channels + self.out_channels, self.out_channels)
    
  def forward(self, x, edge_index):
    # x has shape [N, in_channels]
    # edge_index has shape [2, E]
    #print(f'MPNN x :', x.size())
    return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
    
  def message(self, x_i, x_j):
    # x_i has shape [E, in_channels]
    # x_j has shape [E, in_channels]
    #print('MPNN => xi, xj', x_i.size(), x_j.size())
    tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
    #print('MPNN => messages IN', tmp.size())
    m = self.messages(tmp)
    #print('MPNN => messages OUT', m.size())
    return m
  
  def update(self, aggr_out, x):
    # aggr_out has shape [N, out_channels]
    # x has shape [N, in_channels]
    #print(f'MPNN => x_i', x.size(), ' aggr_out ', aggr_out.size())
    tmp = torch.cat([x, aggr_out], dim=1)
    #print(f'MPNN => tmp', tmp.size())
    return self.update_fn(tmp)