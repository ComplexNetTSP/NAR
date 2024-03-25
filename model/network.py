import torch
from torch import nn
from .encoder import Encoder
from .decoder import Decoder
from .mpnn import MPNN
from torch.functional import F

class Network(nn.Module):
    def __init__(self, latent_dim=128):
        """
        Initializes the Network module.

        Args:
            latent_dim (int, optional): Dimensionality of the latent space. Defaults to 128.
        """
        super(Network, self).__init__()
        self.encoder = Encoder(2, latent_dim)
        self.processor = MPNN(latent_dim*2, latent_dim)
        self.decoder = Decoder(latent_dim, 1)

    def forward(self, batch, max_iter=10):
        """
        Performs forward pass through the network.

        Args:
            batch (Batch): Input batch containing graph data.
            max_iter (int, optional): Maximum number of iterations. Defaults to 10.

        Returns:
            tuple: A tuple containing various loss and accuracy metrics.
        """
        input = torch.stack((batch.pos, batch.s), dim=1).float()
        h = torch.zeros(input.size(0), 128).to(input.device)
        hints_edges = batch.edges_h[1:].to(input.device)
        true_output = batch.edges.to(input.device)
        max_iter = hints_edges.size(0)
        hints_reach = batch.reach_h[1:].to(input.device)

        predictions_edges = torch.zeros(max_iter, batch.edges.size(0)).to(input.device)

        for i in range(max_iter):
            threshold = 0.5

            z = self.encoder(input)
            processor_input = torch.cat([z, h], dim=1)
            h = self.processor(processor_input, batch.edge_index.long()).to(input.device)
            decoder_input = torch.cat((h[batch.edge_index[0]], h[batch.edge_index[1]]), dim=1)
            alpha = self.decoder(decoder_input).view(batch.edges.size(0))

            predictions_edges[i] = alpha.view(batch.edges.size(0))
            predictions_reach = self.calculate_reach(batch, alpha, threshold=0.4)
            
            input = torch.stack((batch.pos.to(input.device), predictions_reach.to(input.device)), dim=1).float()

        predictions_parents = self.get_parent_nodes(batch.edge_index, alpha, batch.s, threshold=0.).to(input.device)
        
        loss_edges = self.calculate_loss(hints_edges, predictions_edges, true_output)
        reach_err = self.reach_accuracy(predictions_reach, batch.reach_h[-1])
        parents_err = self.parents_error(predictions_parents, batch.pi)
        edges_err = self.calculate_percentage_correct(predictions_edges, true_output, max_iter)

        return loss_edges, edges_err, reach_err, parents_err
    
    def calculate_loss(self, hints, predictions, true_output):
        """
        Calculates the loss function based on predicted and true values.

        Args:
            hints (Tensor): Ground truth hints.
            predictions (Tensor): Predicted values.
            true_output (Tensor): True output values.

        Returns:
            tuple: A tuple containing loss values.
        """
        if len(predictions) == 0:
            return torch.tensor(0.0).to(hints.device), torch.tensor(0.0).to(hints.device)
        loss_x = F.binary_cross_entropy(predictions[-1], true_output.type(torch.float))
        loss_h = sum(F.binary_cross_entropy(pred, hint.type(torch.float)) for pred, hint in zip(predictions, hints))
        return loss_x, loss_h

    def reach_accuracy(self, predictions, true_output):
        """
        Calculates the accuracy of reach predictions.

        Args:
            predictions (Tensor): Predicted reach values.
            true_output (Tensor): True reach values.

        Returns:
            float: Accuracy percentage.
        """
        true_output = true_output.float().to(predictions.device)
        predictions = torch.round(predictions).float()
        correct_predictions = torch.sum(predictions == true_output).item()
        total_predictions = true_output.numel()
        accuracy = correct_predictions / total_predictions * 100
        return accuracy

    def calculate_reach(self, graph, alpha, threshold=0.8):
        """
        Calculates reach predictions based on alpha values.

        Args:
            graph (Graph): Input graph data.
            alpha (Tensor): Alpha values.
            threshold (float, optional): Threshold for reach prediction. Defaults to 0.8.

        Returns:
            Tensor: Reach predictions.
        """
        y = torch.zeros((len(graph.s))).to(alpha.device)
        for node_index in range(len(graph.s)):
            connected_edges = torch.logical_or(graph.edge_index[0] == node_index, graph.edge_index[1] == node_index)
            if torch.any(connected_edges):
                alpha_max_proba = alpha[connected_edges].max()
                if alpha_max_proba.item() >= threshold:
                    y[node_index] = 1
        return y

    def get_parent_nodes(self, edge_index, alpha, s, threshold=0.8):
        """
        Determines parent nodes based on alpha values.

        Args:
            edge_index (Tensor): Edge index.
            alpha (Tensor): Alpha values.
            s (Tensor): Tensor of node indices.
            threshold (float, optional): Threshold for parent node determination. Defaults to 0.8.

        Returns:
            Tensor: Parent nodes.
        """
        num_nodes = len(s)
        parent_nodes = torch.arange(num_nodes).to(s.device)
        for node in range(num_nodes):
            incoming_edges = (edge_index[1] == node).nonzero(as_tuple=False).squeeze()
            if incoming_edges.numel() != 0:
                alpha_device = incoming_edges.device
                alpha = alpha.to(alpha_device)
                filtered_edges = incoming_edges[alpha[incoming_edges] >= threshold]
                if filtered_edges.numel() != 0:
                    max_alpha_index = torch.argmax(alpha[filtered_edges])
                    parent_nodes[node] = edge_index[0, filtered_edges[max_alpha_index]].item()
        return parent_nodes

    def parents_error(self, predictions_parents, final_parents):
        """
        Calculates parent nodes error.

        Args:
            predictions_parents (Tensor): Predicted parent nodes.
            final_parents (Tensor): True parent nodes.

        Returns:
            float: Parent nodes error.
        """
        device = predictions_parents.device
        final_parents = final_parents.to(device)
        if len(predictions_parents) == 0 or len(final_parents) == 0:
            return torch.tensor(1.0).to(device)
        return torch.mean(torch.eq(predictions_parents, final_parents).float())
    
    def calculate_percentage_correct(self, predictions, true_output, max_iter, threshold=0.4):
        """
        Calculates the percentage of correct predictions.

        Args:
            predictions (Tensor): Predicted values.
            true_output (Tensor): True output values.
            max_iter (int): Maximum number of iterations.
            threshold (float, optional): Threshold for prediction. Defaults to 0.4.

        Returns:
            float: Percentage of correct predictions.
        """
        correct_predictions = torch.sum((predictions > threshold) == true_output).item()
        total_edges = true_output.size(0) * max_iter
        percentage_correct = correct_predictions / total_edges * 100
        return percentage_correct