import torch 
import torch.nn
from .processor import Processor
from .encoder import Encoder

class EncodeProcessDecode(torch.nn.Module):
  def __init__(self, hidden_channels, aggr):
    super(EncodeProcessDecode, self).__init__()
    input_channel_dim  = 3 * hidden_channels + 1
    self.processor = Processor(input_channel_dim, hidden_channels, aggr=aggr)
    self.encoder = Encoder(hidden_channels)
    
  def stack_hidden(input_hidden, hidden, last_hidden):
    return torch.cat([input_hidden, hidden, last_hidden], dim=-1)
  
  def forward(self, batch):
    # batch[0]: node position 
    # batch[1]: node features
    max_iter = batch.length.item()
    pos = batch.pos
    input_hidden = self.encoder(batch.s)
    hidden = input_hidden
    # iterate over the algorithm steps (example: number of step to perform a BFS)
    for step in range(max_iter):
      last_hidden = hidden
      hidden = self.processor(input_hidden, hidden, last_hidden, edge_index=batch.edge_index, pos=pos)
      # test of the algorithm is finished
      if max_iter == step + 1:
        output_step = self.decoder(self.stack_hidden(input_hidden, hidden, last_hidden))
    return output_step, hidden, pos
