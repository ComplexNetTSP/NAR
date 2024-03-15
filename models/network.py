import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder
from mpnn import MPNN
from torch.functional import F

class Network(nn.Module):
    def __init__(self, latent_dim=128):
        super(Network, self).__init__()
        self.encoder = Encoder(2, latent_dim)
        self.encoder_bn = nn.BatchNorm1d(latent_dim)
        self.processor = MPNN(latent_dim*2, latent_dim)
        self.processor_bn = nn.BatchNorm1d(latent_dim)
        self.decoder = Decoder(latent_dim, 1)

    def forward(self, batch, max_iter=10):
        input = torch.stack((batch.pos, batch.s), dim=1).float()
        h = torch.zeros(input.size(0), 128) # hidden state from the processor
        hints = batch.pi_h[1:] # hints if an edge was passed or not
        true_output = batch.pi # true_output for all the edges if they were passed or not at the end.
        max_iter = hints.size(0)+1
        predictions = torch.zeros(max_iter, batch.pi.size(0))
        predictions_y = torch.zeros(max_iter, batch.s.size(0))
        hints_reach = batch.reach_h[1:] # hints from the reachability
        true_output_reach = batch.reach_h[-1] # true_output expected from the reachability
        for i in range(max_iter):
            z = self.encoder(input) # the encoded input
            #z = self.encoder_bn(z) # batch normalization
            processor_input = torch.cat([z, h], dim=1) # the input to the processor
            h = self.processor(processor_input, batch.edge_index.long()) # the output of the processor
            #h = self.processor_bn(h) # batch normalization
            decoder_input = torch.cat((h[batch.edge_index[0]], h[batch.edge_index[1]]), dim=1)
            alpha = self.decoder(decoder_input).view(batch.pi.size(0))

            predictions[i] = alpha.view(batch.pi.size(0))

            y = torch.zeros((len(batch.s)))
            for node_index in range(0, len(batch.s)):
                alpha_max_proba = alpha[torch.logical_or(batch.edge_index[0] == node_index, batch.edge_index[1] == node_index)].max()
                #print(alpha_max_proba)
                if alpha_max_proba.item() >= 0.8:
                    #print(alpha_max_proba)
                    y[node_index] = 1
            predictions_y[i] = y

            input = torch.stack((batch.pos, y), dim=1).float() # we update the input with the new state

        loss = self.calculate_loss(hints, predictions, true_output)
        y_loss = self.calculate_loss(hints_reach, predictions_y, true_output_reach)
        return y, loss, y_loss
    
    def calculate_loss(self, hints, predictions, true_output):
        loss_x = F.binary_cross_entropy(predictions[-1], true_output.type(torch.float))
        loss_h = 0
        for i in range(hints.size(0)):
            loss_h += F.binary_cross_entropy(predictions[i], hints[i].type(torch.float))
        return loss_x, loss_h