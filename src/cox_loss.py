"""
Cox scikit-survival
https://scikit-survival.readthedocs.io/en/latest/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html
conda install -c sebp scikit-survival
pip install scikit-survival
Cox lifelines
https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html
pip install lifelines
conda install -c conda-forge lifelines

'species' makes matrix inversion problems(high colinearlity) -> should not contain species
"""


import pandas as pd
import numpy as np
import torch
from torch import nn

from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import CoxPHFitter
import pdb

from torch import tensor


# Cox without pacakges
class PartialNLL(nn.Module):
    def __init__(self):
        super(PartialNLL, self).__init__()

    def forward(self, theta, rank, censored):
        """
        theta : features (X) (4 x 34) -> (4 x 1)
        rank : S(ct) derived from y (ct)  (4 x 4)
        censored : event  (4 x 1)
        """
#        theta = theta.double()
#        rank = risk.double()
#        censored = censored.double()

        exp_theta = torch.exp(theta)
        # observed = (censored.size(0) - torch.sum(censored)).cuda()
        observed = 1 - censored
        num_observed = torch.sum(observed)

        # loss = -(torch.sum((theta.reshape(-1)- torch.log(torch.sum((exp_theta * R.t()), 0))) * observed) / num_observed)
        score = rank.T * exp_theta  # 4 x 1  transpose?
        loss = -(torch.sum((theta.view(-1) - torch.log(torch.sum(score, 1))) * observed) / num_observed)
        # pdb.set_trace()

        if np.isnan(loss.data.tolist()):
            for a,b in zip(theta, exp_theta):
                print(a,b)

        return loss

def make_cox_rank(targets_reg):
    y = targets_reg.detach().cpu().unsqueeze(-1)  # ct_true
    
    # define rank (R or S(y))
    rank = np.zeros((y.shape[0], y.shape[0]))
    for i in range(y.shape[0]):
        for j in range(y.shape[0]):
            rank[i,j] = (y[j] >= y[i])

    rank = torch.FloatTensor(rank)
    return rank

#Cox using scikit survival
def Cox(outputs, targets_reg, targets_cla, species):
    result_cox = []
    features = outputs.detach().cpu().squeeze(-1)
    ct_pred  = targets_reg.detach().cpu().unsqueeze(-1)
    event = targets_cla.detach().cpu().unsqueeze(-1)
    
    # try:
    df = pd.DataFrame(np.concatenate([species.numpy(), features.numpy()], axis =1))
    # except:
    #     pdb.set_trace()

    df.columns = ['py', 'tb'] + [ i for i in range(len(features[0]))]
    
    # resolve singular matrix
    if 'rank' in df:
        df = df.drop('rank', axis=1)
    if 'tb' in df:
        df = df.drop('tb', axis=1)
    
    #X = dataframe({species, outputs}), y=array[(event, time)]
    X = df
    y = np.concatenate([event.numpy(), ct_pred.numpy()], axis =1)
    y = np.array([tuple(row) for row in y], dtype = [('event', bool), ('time', float)])

    print(outputs.shape, ' ', X.shape, ' ', y.shape, ' ') #, len(surv_funcs))
    estimator = CoxPHSurvivalAnalysis(alpha=1e-4).fit(X, y)
    surv_funcs = estimator.predict_survival_function(X)
    #stepFunction
    print(len(surv_funcs))
    for fn in surv_funcs:
        # print(fn)
        result_cox = fn(fn.x)
        print(result_cox.shape)
    
    result_cox = np.array(result_cox).astype('float32')
    result_cox = torch.tensor(result_cox, requires_grad = True)

    return result_cox

#Cox_lifelines 
def Cox_lifelines(outputs, targets_reg, targets_cla):
    result_cox = []
    features = outputs.squeeze(-1)
    ct_pred = targets_reg.unsqueeze(-1)
    event = targets_cla.unsqueeze(-1)
    features = features.detach().cpu().numpy()

    df = pd.DataFrame(np.concatenate([ct_pred.numpy(), event.numpy(), features], axis=1))
    
    df.columns = ['ct', 'class'] + [ i for i in range(len(features[0]))]
    
    cph = CoxPHFitter()
    cph.fit(df, duration_col='ct', event_col='class')

    result_cox = cph.predict_log_partial_hazard(df)
    
    result_cox = np.array(result_cox).astype('float32')
    result_cox = torch.tensor(result_cox, requires_grad = True)
    return result_cox