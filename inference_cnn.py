import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import KFold
from src.model_cnn import *
from src.data import *

import argparse
import datetime
from os import listdir
from os.path import isdir, join, basename, dirname


parser = argparse.ArgumentParser(description='Parser for inference.')
parser.add_argument('-k', '--k_folds', type=int, default=5,
                    help='number of folds for cross validation (default: 5)')
parser.add_argument('-b', '--batch_size', type=int, default=20,
                    help='size of batch (default: 20)')
# parser.add_argument('--embedding_dim', type=int, default=24,
#                     help='embedding dimension (default: 24)')
parser.add_argument('--hidden_dim', type=int, default=16,
                    help='hidden dimension (default: 64)')
parser.add_argument('--data_path', default='./data/test_df.csv',
                    help='path for train dataframe (default: ./data/test_df.csv')
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
                    help='max sequence length (default: 434)')

args = parser.parse_args()


if __name__ == '__main__':
    gpu = 0
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    k_folds = args.k_folds
    batch_size = args.batch_size  # same as len(testloader) ?
#     emb_dim = args.embedding_dim
    hidn_dim = args.hidden_dim
    key = args.key
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
        result_name = f'./result/result_{cur_date}.csv'
    else:
        # cur_date = basename(dirname(load_model_path))
        cur_date = load_model_path.split('/')[-1]
        result_name = f'./result/result_{cur_date}.csv'
        
    print(f'Load cnn model at {load_model_path}')
    
    torch.manual_seed(42)
    
    dataset_df = pd.read_csv(args.data_path)
#     dataset_test = dataset_df.to_numpy()
    word2index_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
#     word2index_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'R':4, 'Y':5, 'M':6, 'K':7}
#     with open(args.word_dict,'rb') as f:
#         word2index_dict = pickle.load(f)
    vocab_size = len(word2index_dict)
    
    dataset = Dataset_FRP(dataset_df, key)
    dataset.set_max_seq_len(args.max_len)
    testloader = get_loader_CNN_infer(dataset, batch_size, key, word2index_dict, is_test=True)
    result = []
    
    for fold in list(range(k_folds)):
        model = CNN(args.max_len, vocab_size, hidn_dim, device, args.kernel_size).to(device)
        path_cnn = join(load_model_path, f'model_fold{fold}.pth')
        checkpoint_cnn = torch.load(path_cnn)
        model.load_state_dict(checkpoint_cnn)

        for i, data in enumerate(testloader, 0):
            inputs = data
#             enc_hidn = torch.zeros(1, inputs.shape[0], hidn_dim, device=device)
#             enc_cell = torch.zeros(1, inputs.shape[0], hidn_dim, device=device)
#             outputs, _, _ = encoder(inputs,enc_hidn,enc_cell)
            output = model(inputs.to(device))
            output_cpu = output.detach().cpu().numpy()
            result.append(output_cpu)
            
    mean = np.mean(result, axis=0)
    dataset_df['ct'] = mean
    
    dataset_df.to_csv(result_name, index=0)
    
    print('Inference Completed')
