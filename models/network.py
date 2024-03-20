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
        self.processor = MPNN(latent_dim*2, latent_dim)
        self.decoder = Decoder(latent_dim, 1)

    def forward(self, batch, max_iter=10):
        input = torch.stack((batch.pos, batch.s), dim=1).float()
        h = torch.zeros(input.size(0), 128) # hidden state from the processor
        hints_edges = batch.edges_h[1:] # hints if an edge was passed or not
        true_output = batch.edges # true_output for all the edges if they were passed or not at the end.
        max_iter = hints_edges.size(0)

        hints_reach = batch.reach_h[1:] # hints from the reachability
        
        predictions_edges = torch.zeros(max_iter, batch.edges.size(0))

        for i in range(max_iter):
            threshold = 0.5

            # NETWORK START
            z = self.encoder(input) # the encoded input
            processor_input = torch.cat([z, h], dim=1) # the input to the processor
            h = self.processor(processor_input, batch.edge_index.long()) # the output of the processor
            decoder_input = torch.cat((h[batch.edge_index[0]], h[batch.edge_index[1]]), dim=1)
            alpha = self.decoder(decoder_input).view(batch.edges.size(0))
            # NETWORK END

            # predictions for the edges
            predictions_edges[i] = alpha.view(batch.edges.size(0))
            # update the input with the new state
            predictions_reach = self.calculate_reach(batch, alpha, threshold=0.4)
            
            input = torch.stack((batch.pos, predictions_reach), dim=1).float()

        # predictions for reach 

        # predictions for the parents
        predictions_parents = self.get_parent_nodes(batch.edge_index, alpha, batch.s, threshold=0.)
        
        loss_edges = self.calculate_loss(hints_edges, predictions_edges, true_output)
        loss_reach = self.calculate_reach_loss(predictions_reach, batch.reach_h[-1])
        loss_parents = self.calculate_parents_loss(predictions_parents, batch.pi)
        
        return loss_edges, loss_reach, loss_parents
        
    def calculate_loss(self, hints, predictions, true_output):
        if len(predictions) == 0:
            return torch.tensor(0.0), torch.tensor(0.0)  # Return zero loss if predictions is empty
        loss_x = F.binary_cross_entropy(predictions[-1], true_output.type(torch.float))
        loss_h = 0
        for i in range(hints.size(0)):
            loss_h += F.binary_cross_entropy(predictions[i], hints[i].type(torch.float))
        return loss_x, loss_h
    
    def calculate_reach_loss(self, predictions, true_output):
        return F.binary_cross_entropy(predictions, true_output.type(torch.float))
    

    def calculate_reach(self, graph, alpha, threshold=0.8):
        """
        Calculate reachability values for each node from the alpha values

        Args:
        - graph: Graph object containing graph data
        - alpha: PyTorch tensor containing alpha values
        - threshold: Threshold value for reachability

        Returns:
        - y: PyTorch tensor containing reachability values for each node
        """
        y = torch.zeros((len(graph.s)))
        for node_index in range(len(graph.s)):
            # Check if there are any edges connected to the current node
            connected_edges = torch.logical_or(graph.edge_index[0] == node_index, graph.edge_index[1] == node_index)
            if torch.any(connected_edges):
                alpha_max_proba = alpha[connected_edges].max()
                if alpha_max_proba.item() >= threshold:
                    y[node_index] = 1
        return y


    def get_parent_nodes(self, edge_index, alpha, s, threshold=0.8):
        num_nodes = len(s)
        parent_nodes = torch.arange(num_nodes)  # Initialize parent nodes with their own index

        for node in range(num_nodes):
            # Get all edges that point to the current node
            incoming_edges = (edge_index[1] == node).nonzero(as_tuple=False).squeeze()
            
            # Check if there are any edges pointing to the current node
            if incoming_edges.numel() != 0:
                # Filter alpha based on threshold
                filtered_edges = incoming_edges[alpha[incoming_edges] >= threshold]
                
                # Check if there are any edges left after filtering
                if filtered_edges.numel() != 0:
                    # Get the index of the edge with the highest alpha after filtering
                    max_alpha_index = torch.argmax(alpha[filtered_edges])
                    # Get the parent node
                    parent_nodes[node] = edge_index[0, filtered_edges[max_alpha_index]].item()

        return parent_nodes

    def calculate_parents_loss(self, predictions_parents, final_parents):
        if len(predictions_parents) == 0 or len(final_parents) == 0:
            return torch.tensor(1.0) 
        return 1 - torch.mean(torch.eq(predictions_parents, final_parents).float())