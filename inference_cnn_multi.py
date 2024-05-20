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
from datetime import datetime
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
parser.add_argument('--hidden_dim', type=int, default=256,
                    help='hidden dimension (default: 64)')
parser.add_argument('--data_path', default='./data/210916_train_normalized.csv',
                    help='path for train dataframe (default: ./data/210916_test_normalized.csv')
parser.add_argument('--key', default='F primer',
                    help='sequence type for prediction (default: F primer)')
parser.add_argument('--model_path', default = './model/cnn/211013-134352',
                    help='path for trained encoder (default: latest model)')
parser.add_argument('--word_dict', default='./data/word_dict.pkl',
                    help='path for word dict (default: ./data/word_dict.pkl)')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='conv kernel size (default: 3)')
parser.add_argument('--max_len', type=int, default=40,
                    help='max sequence length (default: 40)')
parser.add_argument('--task', default='reg',
                    help='task type: regression (reg), classification (cls), multi-task learning (multi), ranking (rank)')
parser.add_argument('--infer_cox', type = bool, default = False)

# parser.add_argument('--debug', type=bool, default=False,
#                     help='debug mode (default: False)')

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
#     debug_name = '_debug' if args.debug else ''
    
    # Load recent model
    load_model_path = args.model_path
    # tmp_time = datetime.now().strftime('%y%m%d_%H%M%S')

    if load_model_path is None:
        model_dirs = [d for d in listdir(model_root_path) if isdir(join(model_root_path, d))]
        model_dirs.sort()
        cur_dir_name = model_dirs[-1]
        load_model_path = f'{model_root_path}/{cur_dir_name}/'
    else:
        # cur_dir_name = basename(dirname(load_model_path))
        cur_dir_name = load_model_path.split('/')[-1]

    result_name = f'./result/cnn/{cur_dir_name}/result_{cur_dir_name}.csv'
        
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
    result_cla = []
    result_np = None
    
    for fold in list(range(k_folds)):
        
        model = MultiInputCNN(args.max_len, vocab_size, hidn_dim, device, args.kernel_size).to(device)
        model.eval()
        path_cnn = join(load_model_path, f'model_fold{fold}.pth')
        
        fold_result = []
        fold_result_cla = []
        fold_result_np = None
        
        activation = nn.Sigmoid()

        try:
            checkpoint_cnn = torch.load(path_cnn)
            
            model.load_state_dict(checkpoint_cnn)
            # model.return_hidn_state = True if args.infer_cox else False  # for cox (return before the last layer)

            for i, data in enumerate(testloader, 0):
                # inputs, species = data
                inputs, targets_cla, targets_reg, targets_rank, species = data
                #import pdb; pdb.set_trace()
                if spec == True:
                    outputs = model(inputs, species)
                else:
                    outputs = model(inputs)
                output_cpu = outputs.detach().cpu().numpy()
            
                
                # inference using cox
                # if args.infer_cox == True:
                #     # output : must be in not reduce dim (before the last dense layer)
                #     np_concat = np.concatenate([targets_cla.numpy(), targets_reg.numpy(), targets_rank.numpy(), species.numpy()], axis=1)
                    
                #     if fold_result_np is None:
                #         fold_result_np = np_concat   # result_df should be assigned before loop begins
                #     else:
                #         fold_result_np = np.concatenate([fold_result_np, np_concat])

                   

                # for cls, rank tasks
                outputs_cla = activation(outputs)
                outputs_cla_cpu = outputs_cla.detach().cpu().numpy()
                if args.task == 'rank':
                    outputs_cla_cpu = outputs_cla_cpu * len(dataset)
                
                # result.append(output_cpu)
                fold_result += list(output_cpu)
                fold_result_cla += list(outputs_cla_cpu)
                
        except Exception as e:
            print(e)
            pdb.set_trace()
        
        if len(fold_result) != 0:
            result.append(fold_result)
        if len(fold_result_cla) != 0:
           result_cla.append(fold_result_cla)
        # if fold_result_np is not None and len(fold_result_np) != 0:
        #    result_np = fold_result_np  # use last one
          
    if len(result) == 0:
        raise Exception('No model exists!')

    # infer_cox result save as csv
    elif args.infer_cox == True:
        # result: fold x len(dataset) x hidn_dim
        # mean: len(dataset) x hidn_dim
        pdb.set_trace()
        mean = np.mean(result, axis=0)
        pdb.set_trace()
        
        # result_df: len(dataset) x (hidn_dim + 3)
        pdb.set_trace()
        result_df = pd.DataFrame(np.concatenate([result_np, mean], axis=1), \
                                 columns=['class', 'ct', 'rank', 'py', 'tb'] + [i for i in range(len(mean[0]))])
        print(result_df.head())

        # save as csv
        result_df.to_csv(f'./result/cnn/{cur_dir_name}/output_array_{cur_dir_name}.csv', index= False)
        print(f'Save cox array at: ./result/cnn/{cur_dir_name}/output_array_{cur_dir_name}.csv')

    else:
        # for regression (ct) task 
        mean = np.mean(result, axis=0)
        print(np.squeeze(mean))
        dataset_df['ct_pred'] = mean
        
        # for classification (nan, class) task
        mean_cla = np.mean(result_cla, axis=0)
        if args.task == 'rank':
            mean_cla = mean_cla.round().astype(int)
        print(np.squeeze(mean_cla))
        dataset_df['class_pred'] = mean_cla
        
        # save as csv
        dataset_df.to_csv(result_name, index=0)
    
    print('Inference Completed')
