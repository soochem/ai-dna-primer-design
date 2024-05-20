import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import KFold
from src.model_cnn import *
from src.data_multi_input import *

import argparse
import datetime
from os import listdir
from os.path import isdir, join, basename, dirname

import pdb


parser = argparse.ArgumentParser(description='Parser for inference.')
parser.add_argument('-k', '--k_folds', type=int, default=5,
                    help='number of folds for cross validation (default: 5)')
parser.add_argument('-b', '--batch_size', type=int, default=4,
                    help='size of batch (default: 20)')
# parser.add_argument('--embedding_dim', type=int, default=24,
#                     help='embedding dimension (default: 24)')
parser.add_argument('--species', type=bool, default=True,
                    help='add species feature (default: True)')
parser.add_argument('--hidden_dim', type=int, default=128,
                    help='hidden dimension (default: 64)')
parser.add_argument('--data_path', default='./data/210729_drop.csv',
                    help='path for train dataframe (default: ./data/210729_drop.csv')
parser.add_argument('--key', default='F primer',
                    help='sequence type for prediction (default: F primer)')
parser.add_argument('--model_path', 
                    help='path for trained encoder (default: latest model)')
parser.add_argument('--word_dict', default='./data/word_dict.pkl',
                    help='path for word dict (default: ./data/word_dict.pkl)')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='conv kernel size (default: 3)')
# parser.add_argument('--debug', type=bool, default=False,
#                     help='debug mode (default: False)')
parser.add_argument('--max_len', type=int, default=40,
                    help='max sequence length (default: 40)')
parser.add_argument('--target_name', type=str, default='ct',
                    help='target name to train model (default: ct)')

args = parser.parse_args()


if __name__ == '__main__':
    gpu = 0
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    k_folds = args.k_folds
    batch_size = args.batch_size  # same as len(testloader) ?
#     emb_dim = args.embedding_dim
    hidn_dim = args.hidden_dim
    key = args.key
    spec = args.species
    model_root_path = './model/cnn'
    result_name = f'./result/result.csv'
#     debug_name = '_debug' if args.debug else ''
#     result_name = f'./result/result{debug_name}.csv'
    
    # Load recent model
    load_model_path = args.model_path
    if load_model_path is None:
        model_dirs = [d for d in listdir(model_root_path) if isdir(join(model_root_path, d))]
        model_dirs.sort()
        cur_date = model_dirs[-1]
        load_model_path = f'{model_root_path}/{cur_date}/'
        # result_name = f'./result/result_{cur_date}.csv'
        result_name = join(load_model_path, f'result_{cur_date}.csv')
    else:
        # cur_date = basename(dirname(load_model_path))
        cur_date = load_model_path.split('/')[-1]
        # result_name = f'./result/result_{cur_date}.csv'
        result_name = join(load_model_path, f'result_{cur_date}.csv')
        
    print(f'Load cnn model at {load_model_path}')
    
    torch.manual_seed(42)
    
    dataset_df = pd.read_csv(args.data_path)
#     dataset_test = dataset_df.to_numpy()
    word2index_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
#     word2index_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'R':4, 'Y':5, 'M':6, 'K':7}
#     with open(args.word_dict,'rb') as f:
#         word2index_dict = pickle.load(f)
    vocab_size = len(word2index_dict)
    
    dataset = Dataset_FRP(dataset_df, key, target_name=args.target_name)
    dataset.set_max_seq_len(args.max_len)
    testloader = get_loader_CNN_infer(dataset, batch_size, key, word2index_dict, is_test=True)
    result = []
    
    for fold in list(range(k_folds)):
        model = MultiInputCNN(args.max_len, vocab_size, hidn_dim, device, args.kernel_size).to(device)
        model.eval()
        path_cnn = join(load_model_path, f'model_fold{fold}.pth')
        
        fold_result = np.array([])
        checkpoint_cnn = torch.load(path_cnn)
        
        model.load_state_dict(checkpoint_cnn)
        activation = nn.Sigmoid()

        for i, data in enumerate(testloader, 0):
            inputs, species = data
            if spec == True:
                outputs = model(inputs, species)
            else:
                outputs = model(inputs)
            
            outputs = activation(outputs)
            output_cpu = outputs.detach().cpu().numpy()
            # classifier
            output_cpu = np.where(output_cpu > 0.5, 1, 0)
            # fold_result += list(output_cpu)
            fold_result = np.concatenate((fold_result, output_cpu), axis=0) if fold_result.size else output_cpu

        result.append(fold_result)
    
    if len(result) == 0:
        print(f'No model exists')
    else:
        # print()
        # print(np.asarray(result).shape)
        # print(np.squeeze(result))
        
        # best results from fold (int value is needed)
        # mean = np.mean(result, axis=0)
        med = np.median(result, axis=0).astype(np.int32)
        print(np.squeeze(med))
        dataset_df[args.target_name + '_pred'] = med
        # print(result_name)
        dataset_df.to_csv(result_name, index=0)
        
        len_data = len(dataset_df.label)
        num_normal = sum(dataset_df[args.target_name + '_pred'] == 0)
        num_correct = sum(dataset_df.label == dataset_df[args.target_name + '_pred'])

        print(f'# of 0 prediction (normal ct) : {num_normal}\n' + \
                f'# of correct prediction : {num_correct}\n# of incorrect prediction : {len_data - num_correct}\n' + \
                f'Accuarcy : {round(num_correct/len_data, 2)}')

        # Save no nan data
        no_nan_df = dataset_df[dataset_df[args.target_name + '_pred'] == 0]
        # pdb.set_trace()
        no_nan_df['ct'] = np.where(no_nan_df['ct'].isnull(), 40, no_nan_df['ct'])
        # pdb.set_trace()
        tmp_str = args.data_path.replace('.csv', '')
        no_nan_df.to_csv(f'{tmp_str}_no_nan.csv', index=0)

    
    print('Inference Completed')
