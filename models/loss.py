from torch.functional import F
from torch import nn
import torch

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, batch, batch_pred: torch.Tensor):

        # for every batch find the predicted and true values and send them to the calculate_loss function
        loss_x = 0
        loss_h = 0
        for i in range(batch.len()):
            data = batch[i]
            x = data.reach_h[-1] # true output value
            x_pred = batch_pred[i][-1] # predicted output value
            h_pred = batch_pred[i][:-2] # predicted hint values
            h = data.reach_h[:len(h_pred)] # true hint values
            loss_x += F.binary_cross_entropy(x, x_pred)
            print(loss_x)
            for i in range(h.size(1)):
                loss_h += F.binary_cross_entropy(h[:, i], h_pred[:, i])

        return loss_x, loss_h