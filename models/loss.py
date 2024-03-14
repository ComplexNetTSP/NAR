from torch.functional import F
from torch import nn
import torch

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, hints:torch.tensor, predictions: torch.Tensor, trueOutput: torch.Tensor):

        # for every batch find the predicted and true values and send them to the calculate_loss function
        loss_x = 0
        loss_h = 0
        

        for i in range(hints.size(0)):
            data = batch[i]
            x = trueOutput
            x_pred = batch_pred[i][-1] # predicted output value
            h_pred = batch_pred[i][:-1] # predicted hint values
            h = data[:len(h_pred)] # true hint values
            x = x.type(torch.float32)
            x_pred = x_pred.type(torch.float32)
            loss_x += F.binary_cross_entropy(x_pred, x)
            print(h.size())
            for j in range(h.size(0)):
                loss_h += F.binary_cross_entropy(h[:, j], h_pred[:, j])

        return loss_x, loss_h
