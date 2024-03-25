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
        torch.nn.Linear(self.hidden_channels, self.hidden_channels),  # Linear layer with hidden_channels input and hidden_channels output
        torch.nn.ReLU(),                                              # ReLU activation function
        torch.nn.Linear(self.hidden_channels, self.hidden_channels)   # Linear layer with hidden_channels input and self.hidden_channels output
    )
    
  def forward(self, x, edge_index):
      device = next(self.parameters()).device  # Get the device of the model parameters
      x = x.to(device)
      edge_index = edge_index.to(device)

      out = self.propagate(edge_index, x=x)
      out = self.mlp(out)
      if self.activation is not None:
          out = self.activation(out)
      return out
    
  def message(self, x_i, x_j):
    tmp = torch.cat([x_i, x_j], dim=1) 
    m = self.messages(tmp)
    return m
