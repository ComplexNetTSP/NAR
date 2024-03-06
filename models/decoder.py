from torch import nn
class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.lin(x)