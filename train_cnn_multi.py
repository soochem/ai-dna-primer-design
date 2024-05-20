from __future__ import unicode_literals, print_function, division
import random

import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold

from src.model_cnn import *
from src.data_multi_input import *
from src.plot_utils import show_plot
from src.pointwise_rank import *
from src.cox_loss import *

import argparse
import datetime
from os import makedirs

from src.callbacks import EarlyStopping
# check output
import scipy
# import sklearn
# import pdb 


parser = argparse.ArgumentParser(description='Parser for training.')
parser.add_argument('-k', '--k_folds', type=int, default=5,
                    help='number of folds for cross validation (default: 5)')
parser.add_argument('-e', '--num_epochs', type=int, default=50,
                    help='number of epochs (default: 500)')
parser.add_argument('-b', '--batch_size', type=int, default=4,
                    help='size of batch (default: 10)')
parser.add_argument('-l', '--learning_rate', type=float, default=5e-6,
                    help='learning rate (default: 5e-5)')
# parser.add_argument('-t', '--teacher_forcing_rate', type=float, default=0.5,
#                     help='teacher forcing rate (default: 0.5)')
parser.add_argument('-d', '--dropout', type=float, default=0.3,
                    help='encoder dropout rate (default: 0.5)')
# parser.add_argument('--embedding_dim', type=int, default=24,
#                     help='embedding dimension (default: 24)')
parser.add_argument('--species', default=True,
                    help='add species feature (default: True)')
parser.add_argument('--hidden_dim', type=int, default=256,
                    help='hidden dimension (default: 64)')
parser.add_argument('--plot_every', type=int, default=1,
                    help='number of epochs for plotting (default: 10)')
parser.add_argument('--print_every', type=int, default=1,
                    help='number of epochs for printing losses for plot (default: 1)')
parser.add_argument('--data_path', default='./data/210916_train_normalized.csv',
                    help='path for train dataframe (default: ./data/train_df.csv')
parser.add_argument('--key', default='R primer',
                    help='sequence type for prediction (default: R primer)')
parser.add_argument('--word_dict', default='./data/word_dict.pkl',
                    help='path for word dict (default: ./data/word_dict.pkl)')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='conv kernel size (default: 3)')
# parser.add_argument('--debug', type=bool, default=False,
#                     help='debug mode (default: False)')
parser.add_argument('--gpu', type=int, default=1,
                    help='gpu assigned (default: 0)')
parser.add_argument('--patience', type=int, default=20,
                    help='gpu assigned (default: 20)')
parser.add_argument('--target_name', type=str, default='ct',
                    help='target name to train model (default: ct)')
parser.add_argument('--loss_function', type=str, default='rank_loss_multi',
                    help='loss function to train model (default: mse_loss)')

args = parser.parse_args()
cur_date = datetime.datetime.now().strftime('%y%m%d-%H%M%S')


if __name__ == '__main__':
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    k_folds = args.k_folds
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
#     plot_every = args.plot_every
#     emb_dim = args.embedding_dim
    hidn_dim = args.hidden_dim
    key = args.key
    spec = args.species
    max_seq_len = 40
    
    result_img_name = 'train'
    # save_result_path = f'./model/cnn/{cur_date}/'
    save_model_path = f'./model/cnn/e{num_epochs}-b{batch_size}-lr{learning_rate}-h{hidn_dim}-f{k_folds}/'
    # save_result_path = f'./result/cnn/{cur_date}/'
    save_result_path = f'./result/cnn/e{num_epochs}-b{batch_size}-lr{learning_rate}-h{hidn_dim}-f{k_folds}/'
    
    # Experiment results (loss)
    train_results = {}
    val_results = {}
    train_results_reg = {}
    val_results_reg = {}
    train_results_rank = {}
    val_results_rank = {}
    
    torch.manual_seed(42)
    
    # Load datasets
    dataset_train = pd.read_csv(args.data_path)  #.to_numpy()
#     with open(args.word_dict,'rb') as f:
#         word2index_dict = pickle.load(f)
#     word2index_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'R':4, 'Y':5, 'M':6, 'K':7}
    word2index_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    vocab_size = len(word2index_dict)
    
    dataset = Dataset_FRP(dataset_train, key)
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    # Make directory for saved model
    try:
        makedirs(save_model_path, exist_ok=True)
        makedirs(save_result_path, exist_ok=True)
    except:
        print("Error while making directories")
        raise

    print('Saved Model Path: %s' % save_model_path)

    print('----------------------------------')

    for fold, (train_index, dev_index) in enumerate(kfold.split(dataset)):
        
        print(f'Fold {fold}')
        print('----------------------------------')
        
        save_path = save_model_path + f'/model_fold{fold}.pth'
        # parameters for printing
        train_loss_plot_list = []
        train_loss_plot = 0.0
        valid_loss_plot_list = []
        valid_loss_plot = 0.0
        # avg losses per epoch
        avg_train_losses = []
        avg_train_losses_reg = []
        avg_train_losses_rank = []
        avg_val_losses = []
        # correlation
        corr_plot_list = []
        # early stop
        early_stop_pat = args.patience
        early_stop_cnt = 0
        best_train_loss = np.inf
        best_valid_loss = np.inf

        # data
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        dev_subsampler = torch.utils.data.SubsetRandomSampler(dev_index)
    
        trainloader = get_loader_CNN(dataset, train_subsampler, batch_size, key, word2index_dict)
        devloader = get_loader_CNN(dataset, dev_subsampler, batch_size, key, word2index_dict)
        
        # model
        model = MultiInputCNN(max_seq_len, vocab_size, hidn_dim, device, args.kernel_size, args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        if args.loss_function == 'bce_loss':
            loss_function = nn.BCELoss()
            activation = nn.Sigmoid()
        elif args.loss_function == 'mse_loss':
            loss_function = nn.MSELoss()
        elif args.loss_function == 'multi':
            loss_function = nn.BCELoss()
            loss_function_reg = nn.MSELoss()
            activation = nn.Sigmoid()
        elif args.loss_function == 'rank_loss':
            loss_function = RankLoss()
            activation = nn.Sigmoid()
        elif args.loss_function == 'rank_loss_multi':
            loss_function = nn.BCELoss()
            loss_function_reg = RankLoss()
            activation = nn.Sigmoid()
        elif args.loss_function == 'marginrank_loss':
            loss_function = nn.MarginRankingLoss()
        elif args.loss_function == 'marginrank_multi':
            loss_function = nn.BCELoss()
            loss_function_reg = nn.MarginRankingLoss()
            activation = nn.Sigmoid()
            #loss_function_reg = nn.MSELoss()
        elif args.loss_function == 'all':
            loss_function = nn.BCELoss()
            loss_function_rank = nn.MarginRankingLoss()
            loss_function_reg = nn.MSELoss()
            activation = nn.Sigmoid()
        elif args.loss_function == 'cox_loss':
            loss_function = PartialNLL()
        else:
            print('Check Loss')

        for epoch in range(1, num_epochs+1):
            
            # losses per epoch while training/evaluating
            train_loss = 0.0
            valid_loss = 0.0
            train_losses = []
            valid_losses = []           
            train_loss_reg = 0.0
            valid_loss_reg = 0.0
            train_losses_reg = []
            valid_losses_reg = []
            train_loss_rank = 0.0
            valid_loss_rank = 0.0
            train_losses_rank = []
            valid_losses_rank = []
            # correlation
            corr_list = []
            
            # train
            model.train()
            # model.return_hidn_state = True if args.loss_function=='cox_loss' else False

            for i, data in enumerate(trainloader, 0):
                inputs, targets_cla, targets_reg, targets_rank, species = data
                if spec == True:
                    outputs = model(inputs, species)
                else:
                    outputs = model(inputs)

                #print(outputs)
                # if args.loss_function != 'cox_loss':
                #     # cox output : must be in not reduce dim (before the last dense layer)
                #     pdb.set_trace()
                outputs = outputs.view(-1).to(device)
                targets_cla = targets_cla.type(torch.FloatTensor).view(-1).to(device)
                targets_reg = targets_reg.type(torch.FloatTensor).view(-1).to(device)
                targets_rank = targets_rank.type(torch.FloatTensor).view(-1).to(device)
                

                # print(outputs.size())
                # print(outputs)
                # idx = [i for i in range(len(targets_cla)) if targets_cla[i] == 1]
                idx = [i for i in range(len(targets_cla)) if targets_cla[i] == 0]
                if args.loss_function == 'bce_loss':
                    losses = loss_function(activation(outputs), targets_cla)
                elif args.loss_function == 'mse_loss':
                    if len(idx) == 0:
                        losses = torch.zeros(1, requires_grad=True).to(device)
                    else:
                        losses = loss_function(outputs[idx], targets_reg[idx])
                elif args.loss_function == 'multi':
                    losses = loss_function(activation(outputs), targets_cla)
                    if len(idx) == 0:
                        losses_reg = torch.zeros(1).to(device)
                    else:
                        losses_reg = loss_function_reg(outputs[idx], targets_reg[idx])
                elif args.loss_function == 'rank_loss':
                    losses = loss_function(activation(outputs), targets_rank, len(dataset))
                elif args.loss_function == 'rank_loss_multi':
                    losses = loss_function(activation(outputs), targets_cla)
                    losses_reg = loss_function_reg(activation(outputs), targets_rank, len(dataset))
                elif args.loss_function == 'marginrank_loss':
                    if len(idx) >= 2:
                        a, b, c = MarginRank(outputs[idx],targets_reg[idx])
                        losses = loss_function(a.to(device), b.to(device), c.to(device))
                elif args.loss_function == 'marginrank_multi':
                    losses = loss_function(activation(outputs), targets_cla)
                    if len(idx) >= 2:
                        a, b, c = MarginRank(outputs[idx],targets_reg[idx])
                        losses_reg = loss_function_reg(a.to(device), b.to(device), c.to(device))
                    else:
                        losses_reg = torch.zeros(1, requires_grad=True).to(device)
                elif args.loss_function == 'all':
                    losses = loss_function(activation(outputs), targets_cla)
                    if len(idx) >= 2:
                        a, b, c = MarginRank(outputs[idx],targets_reg[idx])
                        losses_rank = loss_function_rank(a.to(device), b.to(device), c.to(device))
                    else:
                        losses_rank = torch.zeros(1, requires_grad=True).to(device)
                    if len(idx) == 0:
                        losses_reg = torch.zeros(1, requires_grad=True).to(device)
                    else:
                        losses_reg = loss_function_reg(outputs[idx], targets_reg[idx])
                elif args.loss_function == 'cox_loss':
                    rank = make_cox_rank(targets_reg)
                    losses = loss_function(outputs.to(device), rank.to(device), targets_cla.to(device))
                else:
                    print('Check Loss')

                if 'multi' in args.loss_function:
                    losses_tot = 100*losses+losses_reg
                    losses_tot.backward(retain_graph=True)
                elif 'all' in args.loss_function:
                    losses_tot = losses+losses_reg+losses_rank
                    losses_tot.backward(retain_graph=True)
                else:
                    losses = losses
                    losses.backward(retain_graph=True)
                    #pdb.set_trace()
                optimizer.step()
                
                
                train_loss += losses.item()
                train_losses.append(losses.item())
                
                if 'multi' in args.loss_function:
                    train_loss_reg += losses_reg.item()
                    train_losses_reg.append(losses_reg.item())
                if 'all' in args.loss_function:
                    train_loss_reg += losses_reg.item()
                    train_losses_reg.append(losses_reg.item())
                    train_loss_rank += losses_rank.item()
                    train_losses_rank.append(losses_rank.item())
                
                if len(outputs[idx]) >= 3:
                    cor, p = scipy.stats.pearsonr(outputs[idx].cpu().detach().numpy(), targets_reg[idx].cpu().detach().numpy())
                    corr_list.append(cor)

            # evaluate
            model.eval()

            for i, data in enumerate(devloader, 0):
                inputs, targets_cla, targets_reg, targets_rank, species = data
                if spec == True:
                    outputs = model(inputs, species)
                else:
                    outputs = model(inputs)
                outputs = outputs.view(-1).to(device)
                targets_cla = targets_cla.type(torch.FloatTensor).view(-1).to(device)
                targets_reg = targets_reg.type(torch.FloatTensor).view(-1).to(device)
                targets_rank = targets_rank.type(torch.FloatTensor).view(-1).to(device)

                idx = [i for i in range(len(targets_cla)) if targets_cla[i] == 1]
                if args.loss_function == 'bce_loss':
                    losses = loss_function(activation(outputs), targets_cla)
                elif args.loss_function == 'mse_loss':
                    if len(idx) == 0:
                        losses = torch.zeros(1).to(device)
                    else:
                        losses = loss_function(outputs[idx], targets_reg[idx])
                elif args.loss_function == 'multi':
                    losses = loss_function(activation(outputs), targets_cla)
                    if len(idx) == 0:
                        losses_reg = torch.zeros(1).to(device)
                    else:
                        losses_reg = loss_function_reg(outputs[idx], targets_reg[idx])
                elif args.loss_function == 'rank_loss':
                    losses = loss_function(activation(outputs), targets_rank)
                elif args.loss_function == 'rank_loss_multi':
                    losses = loss_function(activation(outputs), targets_cla)
                    losses_reg = loss_function_reg(activation(outputs), targets_rank)
                elif args.loss_function == 'marginrank_loss':
                    if len(idx) >= 2:
                        a, b, c = MarginRank(outputs[idx],targets_reg[idx])
                        losses = loss_function(a.to(device), b.to(device), c.to(device))
                elif args.loss_function == 'marginrank_multi':
                    losses = loss_function(activation(outputs), targets_cla)
                    if len(idx) >= 2:
                        a, b, c = MarginRank(outputs[idx],targets_reg[idx])
                        losses_reg = loss_function_reg(a.to(device), b.to(device), c.to(device))
                    else:
                        losses_reg = torch.zeros(1).to(device)
                elif args.loss_function == 'all':
                    losses = loss_function(activation(outputs), targets_cla)
                    if len(idx) >= 2:
                        a, b, c = MarginRank(outputs[idx],targets_reg[idx])
                        losses_rank = loss_function_rank(a.to(device), b.to(device), c.to(device))
                    if len(idx) == 0:
                        losses_reg = torch.zeros(1).to(device)
                    else:
                        losses_reg = loss_function_reg(outputs[idx], targets_reg[idx])
                elif args.loss_function == 'cox_loss':
                    rank = make_cox_rank(targets_reg)
                    losses = loss_function(outputs.to(device), rank.to(device), targets_cla.to(device))
                else:
                    print('Check Loss')

                valid_loss += losses.item()
                valid_losses.append(losses.item())
                if 'multi' in args.loss_function:
                    valid_loss_reg += losses_reg.item()
                    valid_losses_reg.append(losses_reg.item())
                if 'all' in args.loss_function:
                    valid_loss_reg += losses_reg.item()
                    valid_losses_reg.append(losses_reg.item())
                    valid_loss_rank += losses_reg.item()
                    valid_losses_rank.append(losses_reg.item())
                    
            # avg(loss) per epoch
            avg_train_loss = np.average(train_losses)
            avg_train_losses.append(avg_train_loss)
            avg_val_loss = np.average(valid_losses)
            
            if 'multi' in args.loss_function:
                avg_train_loss_reg = np.average(train_losses_reg)
                avg_train_losses_reg.append(avg_train_loss_reg)
                avg_val_loss_reg = np.average(valid_losses_reg)
            if 'all' in args.loss_function:
                avg_train_loss_reg = np.average(train_losses_reg)
                avg_train_losses_reg.append(avg_train_loss_reg)
                avg_val_loss_reg = np.average(valid_losses_reg)
                avg_train_loss_rank = np.average(train_losses_rank)
                avg_train_losses_rank.append(avg_train_loss_rank)
                avg_val_loss_rank = np.average(valid_losses_rank)
                
            # early stop
            if len(avg_val_losses) != 0 and avg_val_loss >= avg_val_losses[-1]:
                early_stop_cnt += 1
            else:
                early_stop_cnt = 0
                best_train_loss = avg_train_loss
                best_valid_loss = avg_val_loss
                if 'multi' in args.loss_function:
                    best_train_loss_reg = avg_train_loss_reg
                    best_valid_loss_reg = avg_val_loss_reg
                if 'all' in args.loss_function:
                    best_train_loss_reg = avg_train_loss_reg
                    best_valid_loss_reg = avg_val_loss_reg
                    best_train_loss_rank = avg_train_loss_rank
                    best_valid_loss_rank = avg_val_loss_rank
                torch.save(model.state_dict(), save_path)
            avg_val_losses.append(avg_val_loss)
            if 'multi' in args.loss_function:
                avg_val_losses.append(avg_val_loss+avg_val_loss_reg)
            if 'all' in args.loss_function:
                avg_val_losses.append(avg_val_loss+avg_val_loss_reg+avg_val_loss_rank)
            
            ## add
            train_loss_plot += avg_train_loss
            valid_loss_plot += avg_val_loss
            if 'multi' in args.loss_function:
                train_loss_plot += avg_train_loss_reg
                valid_loss_plot += avg_val_loss_reg
            if 'all' in args.loss_function:
                train_loss_plot += avg_train_loss_reg
                valid_loss_plot += avg_val_loss_reg
                train_loss_plot += avg_train_loss_rank
                valid_loss_plot += avg_val_loss_rank

            if early_stop_cnt >= early_stop_pat:
                break
            
            avg_corr = np.average(corr_list)
            corr_plot_list.append(avg_corr)
            #import pdb; pdb.set_trace()

            if epoch % args.print_every == 0:
                if 'multi' in args.loss_function:
                    print('Epoch %d / %d (%d%%) train loss cla: %.4f, train loss reg: %.4f, valid loss cla: %.4f, valid loss reg: %.4f, correlation: %.4f' \
                    % (epoch, num_epochs, epoch / num_epochs * 100, avg_train_loss, avg_train_loss_reg, avg_val_loss, avg_val_loss_reg, avg_corr))
                elif 'all' in args.loss_function:
                    print('Epoch %d / %d (%d%%) train cla: %.4f, train rank: %.4f, train reg: %.4f, valid cla: %.4f, valid rank: %.4f, valid reg: %.4f, correlation: %.4f' \
                    % (epoch, num_epochs, epoch / num_epochs * 100, avg_train_loss, avg_train_loss_rank, avg_train_loss_reg, avg_val_loss, avg_val_loss_rank, avg_val_loss_reg, avg_corr))
                else:
                    print('Epoch %d / %d (%d%%) train loss: %.4f, valid loss: %.4f, correlation: %.4f' \
                    % (epoch, num_epochs, epoch / num_epochs * 100, avg_train_loss, avg_val_loss, avg_corr))

            # Plot
            if epoch % args.plot_every == 0:
                avg_train_loss_plot = train_loss_plot / (args.plot_every)  #*(i+1))
                avg_valid_loss_plot = valid_loss_plot / (args.plot_every)  #*(i+1))
                train_loss_plot_list.append(avg_train_loss_plot)
                train_loss_plot = 0.0
                valid_loss_plot_list.append(avg_valid_loss_plot)
                valid_loss_plot = 0.0
        
        # train_results[fold] = train_loss/(len(trainloader))
        train_results[fold] = best_train_loss
        val_results[fold] = best_valid_loss
        if 'multi' in args.loss_function:
            train_results_reg[fold] = best_train_loss_reg
            val_results_reg[fold] = best_valid_loss_reg
        if 'all' in args.loss_function:
            train_results_reg[fold] = best_train_loss_reg
            val_results_reg[fold] = best_valid_loss_reg
            train_results_rank[fold] = best_train_loss_rank
            val_results_rank[fold] = best_valid_loss_rank
        
        show_plot(train_loss_plot_list, args.plot_every, fold, \
                    eval_points=valid_loss_plot_list, save_path=save_result_path, file_name=result_img_name)
        
        print('-----------------------------------')
        print(f'Fold {fold} Training Loss: {train_results[fold]}')
        print('Average Training Loss: %.4f' % (sum(train_results.values())/len(train_results.items())))
        
        # Save model
        # save_path = save_model_path + f'/model_fold{fold}.pth'
        # torch.save(model.state_dict(), save_path)
        checkpoint_cnn = torch.load(save_path)
        model.load_state_dict(checkpoint_cnn)
        
        print('-------- Starting Evaluation --------')
        
        val_loss = 0.0
        val_loss_reg = 0.0
        val_loss_rank = 0.0
        model.eval()
        with torch.no_grad():
                        
            for i, data in enumerate(devloader, 0):
                inputs, targets_cla, targets_reg, targets_rank, species = data
                if spec == True:
                    outputs = model(inputs, species)
                else:
                    outputs = model(inputs)
                outputs = outputs.view(-1).to(device)
                targets_cla = targets_cla.type(torch.FloatTensor).view(-1).to(device)
                targets_reg = targets_reg.type(torch.FloatTensor).view(-1).to(device)
                targets_rank = targets_rank.type(torch.FloatTensor).view(-1).to(device)

                idx = [i for i in range(len(targets_cla)) if targets_cla[i] == 1]
                if args.loss_function == 'bce_loss':
                    losses = loss_function(activation(outputs), targets_cla)
                elif args.loss_function == 'mse_loss':
                    if len(idx) == 0:
                        losses = torch.zeros(1).to(device)
                    else:
                        losses = loss_function(outputs[idx], targets_reg[idx])
                elif args.loss_function == 'multi':
                    losses = loss_function(activation(outputs), targets_cla)
                    if len(idx) == 0:
                        losses_reg = torch.zeros(1).to(device)
                    else:
                        losses_reg = loss_function_reg(outputs[idx], targets_reg[idx])
                elif args.loss_function == 'rank_loss':
                    losses = loss_function(activation(outputs), targets_rank)
                elif args.loss_function == 'rank_loss_multi':
                    losses = loss_function(activation(outputs), targets_cla)
                    losses_reg = loss_function_reg(activation(outputs), targets_rank)
                elif args.loss_function == 'marginrank_loss':
                    if len(idx) >= 2:
                        a, b, c = MarginRank(outputs[idx],targets_reg[idx])
                        losses = loss_function(a.to(device), b.to(device), c.to(device))
                elif args.loss_function == 'marginrank_multi':
                    losses = loss_function(activation(outputs), targets_cla)
                    if len(idx) >= 2:
                        a, b, c = MarginRank(outputs[idx],targets_reg[idx])
                        losses_reg = loss_function_reg(a.to(device), b.to(device), c.to(device))
                    else:
                        losses_reg = torch.zeros(1).to(device)
                elif args.loss_function == 'all':
                    losses = loss_function(activation(outputs), targets_cla)
                    if len(idx) >= 2:
                        a, b, c = MarginRank(outputs[idx],targets_reg[idx])
                        losses_rank = loss_function_rank(a.to(device), b.to(device), c.to(device))
                    else:
                        losses_rank = torch.zeros(1).to(device)
                    if len(idx) == 0:
                        losses_reg = torch.zeros(1).to(device)
                    else:
                        losses_reg = loss_function_reg(outputs[idx], targets_reg[idx])
                elif args.loss_function == 'cox_loss':
                    rank = make_cox_rank(targets_reg)
                    losses = loss_function(outputs.to(device), rank.to(device), targets_cla.to(device))
                else:
                    print('Check Loss')

                val_loss += losses.item()
                if 'multi' in args.loss_function:
                    val_loss_reg += losses_reg.item()
                if 'all' in args.loss_function:
                    val_loss_reg += losses_reg.item()
                    val_loss_rank += losses_rank.item()

            print('val_loss of fold: %.4f' % (val_loss/len(devloader)))
            print('-----------------------------------')
            val_results[fold] = val_loss/len(devloader)
            if 'multi' in args.loss_function:
                print('val_loss cla of fold: %.4f' % (val_loss/len(devloader)))
                print('val_loss reg of fold: %.4f' % (val_loss_reg/len(devloader)))
                print('-----------------------------------')
                val_results[fold] = val_loss/len(devloader)
                val_results_reg[fold] = val_loss_reg/len(devloader)
            if 'all' in args.loss_function:
                print('val_loss cla of fold: %.4f' % (val_loss/len(devloader)))
                print('val_loss reg of fold: %.4f' % (val_loss_reg/len(devloader)))
                print('val_loss rank of fold: %.4f' % (val_loss_rank/len(devloader)))
                print('-----------------------------------')
                val_results[fold] = val_loss/len(devloader)
                val_results_reg[fold] = val_loss_reg/len(devloader)
                val_results_rank[fold] = val_loss_rank/len(devloader)
        
        print(f'K-Fold CV Results of {k_folds} Folds')
        print('-----------------------------------')
        
        if 'multi' in args.loss_function:
            for f in range(fold+1):
                print('[Fold %d] train loss cla: %.4f, train loss reg: %.4f, valid loss cla: %.4f, valid loss reg: %.4f' %\
                    (f, train_results[f], train_results_reg[f], val_results[f], val_results_reg[f]))
            print('%d folds average train loss cla: %.4f, train loss reg: %.4f, valid loss cla: %.4f, valid loss reg: %.4f' \
                % ((fold+1), sum(train_results.values())/len(train_results.items()), sum(train_results_reg.values())/len(train_results_reg.items()),
                   sum(val_results.values())/len(val_results.items()), sum(val_results_reg.values())/len(val_results_reg.items())))
        elif 'all' in args.loss_function:
            for f in range(fold+1):
                print('[Fold %d] train cla: %.4f, train rank: %.4f, train reg: %.4f, valid cla: %.4f, valid rank: %.4f, valid reg: %.4f' %\
                    (f, train_results[f], train_results_rank[f], train_results_reg[f], val_results[f], val_results_rank[f], val_results_reg[f]))
            print('%d folds average train cla: %.4f, train rank: %.4f, train reg: %.4f, valid cla: %.4f, valid rank: %.4f, valid reg: %.4f' \
                % ((fold+1), sum(train_results.values())/len(train_results.items()), sum(train_results_rank.values())/len(train_results_rank.items()),sum(train_results_reg.values())/len(train_results_reg.items()),
                   sum(val_results.values())/len(val_results.items()), sum(val_results_rank.values())/len(val_results_rank.items()), sum(val_results_reg.values())/len(val_results_reg.items())))
        else:
            for f in range(fold+1):
                print('[Fold %d] train loss: %.4f, valid loss: %.4f' %\
                (f, train_results[f], val_results[f]))
            print('%d folds average train loss: %.4f, valid loss: %.4f' \
            % ((fold+1), sum(train_results.values())/len(train_results.items()),
               sum(val_results.values())/len(val_results.items())))
        print('Saved Model Path: %s' % save_model_path)
        print('-----------------------------------')


