import torch

import pdb
from itertools import combinations

class RankLoss():
    def __init__(self):
        self.padded_value_indicator = -1
        # self.no_of_levels = len(y_pred)
        # self.y_pred = y_pred
        # self.y_true = y_true 
    
    def pointwise_rmse(self, y_pred, y_true, no_of_levels, padded_value_indicator):
        """
        Pointwise RMSE loss.
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param no_of_levels: number of unique ground truth values
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :return: loss value, a torch.Tensor
        
        https://github.com/allegro/allRank/blob/68ba8e3929929d705497d44705cbfdd7edce6f5c/allrank/models/losses/pointwise.py  
        """
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        mask = y_true == padded_value_indicator
        valid_mask = (y_true != padded_value_indicator).type(torch.float32)

        y_true[mask] = 0
        y_pred[mask] = 0

        errors = (y_true - no_of_levels * y_pred)
        squared_errors = errors ** 2

        mean_squared_errors = torch.sum(squared_errors, dim=-1) / torch.sum(valid_mask, dim=-1)
        rmses = torch.sqrt(mean_squared_errors)

        return torch.mean(rmses)

    # def inner_train(self, y_pred, y_true, no_of_levels=None):
    def __call__(self, y_pred, y_true, no_of_levels=None):
        if no_of_levels is None:
            # torch max TODO
            no_of_levels = torch.max(y_true)

        losses = self.pointwise_rmse(y_pred, y_true, no_of_levels, self.padded_value_indicator)
        return losses
    
def MarginRank(output, target):
    comb = combinations(output, 2)
    idx = combinations(range(len(target)), 2)
    input1 = []
    input2 = []
    targets = []
    for in1, in2 in comb:
        input1.append(in1)
        input2.append(in2)
    for idx1, idx2 in idx:
        if target[idx1] > target[idx2]:
            targets.append(1.)
        else:
            targets.append(-1.)
    input1 = torch.tensor(input1,requires_grad=True)
    input2 = torch.tensor(input2,requires_grad=True)
    targets = torch.tensor(targets)
    return input1, input2, targets
