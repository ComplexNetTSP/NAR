import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim = 128):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lin = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.lin(x)