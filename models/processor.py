import torch.nn as nn
import torch
import torch_geometric.nn as pyg_nn
from .gin import gin_module

class Processor_legacy(nn.Module):
  def __init__(self, in_channels, out_channels, aggr, layer_norm=False):
    super().__init__()
    self.layer_norm = layer_norm
    self.processor = gin_module(in_channels=in_channels, out_channels=out_channels, aggr=aggr)
    if self.layer_norm:
      self.norm = pyg_nn.LayerNorm(out_channels, mode='node')
  
  def stack_hidden(self, input_hidden, hidden, last_hidden, pos):
    return torch.cat([input_hidden, hidden, last_hidden, pos.unsqueeze(1)], dim=-1)
  
  def forward(self, input_hidden, hidden, last_hidden, edge_index, pos):
    stacked = self.stack_hidden(input_hidden, hidden, last_hidden, pos)
    out = self.processor(stacked, edge_index)
    if self.layer_norm:
      out = self.norm(out)
    return out
