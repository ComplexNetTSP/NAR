from torch import nn
import torch
class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lin = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2=None):
        if x2 is not None:
            concatenated_input = torch.cat((x1, x2), dim=1)
            return self.lin(concatenated_input)
        else:
            return self.sigmoid(self.lin(x1))