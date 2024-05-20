import argparse
from datetime import datetime
from os import listdir
from os.path import isdir, join, basename, dirname

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

import logging

import pdb


parser = argparse.ArgumentParser(description='Parser for inference.')
parser.add_argument('--model_path', default = './model/cnn/211005-102928',
                    help='path for trained encoder (default: latest model)')
parser.add_argument('--train_cox_data_path', default = 'output_array_211012_225058.csv',
                    help='path for trained encoder (default: latest model)')
parser.add_argument('--test_cox_data_path', default = 'output_array_211012_225058.csv',
                    help='path for trained encoder (default: latest model)')
parser.add_argument('--test_data_path', default = '210916_test_drop.csv',
                    help='path for trained encoder (default: latest model)')
parser.add_argument('--recover_norm_val', type=bool, default = True,
                    help='recover norm by mean and variance')
                    
args = parser.parse_args()

# Load recent model
load_model_path = args.model_path
model_root_path = './model/cnn'
# tmp_time = datetime.now().strftime('%y%m%d_%H%M%S')

if load_model_path is None:
    model_dirs = [d for d in listdir(model_root_path) if isdir(join(model_root_path, d))]
    model_dirs.sort()
    cur_dir_name = model_dirs[-1]
    load_model_path = f'{model_root_path}/{cur_dir_name}/'
else:
    # cur_dir_name = basename(dirname(load_model_path))
    cur_dir_name = load_model_path.split('/')[-1]
# result_name = f'./result/cnn/{cur_dir_name}/result_{tmp_time}.csv'

np_train_name = f'./result/cnn/{cur_dir_name}/{args.train_cox_data_path}'
np_test_name = f'./result/cnn/{cur_dir_name}/{args.test_cox_data_path}'
cox_result_name = f'./result/cnn/{cur_dir_name}/cox_result.csv'

train_df = pd.read_csv(np_train_name)
test_df = pd.read_csv(np_test_name)

# TODO resolve NaN
# result_df = result_df.dropna(subset=['ct'])
train_df = train_df.drop('rank', axis=1)
test_df = test_df.drop('rank', axis=1)
train_df = train_df.drop('tb', axis=1)
test_df = test_df.drop('tb', axis=1)

# Cox regression
# Ref. https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html
cph = CoxPHFitter()

cph.fit(train_df, duration_col='ct', event_col='class')

logging.info(cph.print_summary())  # access the individual results using cph.summary

# TODO save as txt
# with open(np_name.replace('output_array', 'cox_summary'), 'w') as f:
#     f.write(summary)


result_cox = cph.predict_log_partial_hazard(test_df)
result_cox = pd.DataFrame(result_cox, columns=['ct_pred'])
result_cox['ct_true'] = test_df['ct']

if args.recover_norm_val:
    mean_var = pd.read_csv('./data/210916_mean_var.csv', index_col=0)
    result_cox['ct_pred_recover'] = np.where(test_df['py']==1.0,
                result_cox['ct_pred'] * (mean_var['pyogenes_var']['ct'])**(1/2) + mean_var['pyogenes_mean']['ct'],
                result_cox['ct_pred'] * (mean_var['tuberculosis_var']['ct'])**(1/2) + mean_var['tuberculosis_mean']['ct'])
    
    result_cox['ct_true_recover'] = np.where(test_df['py']==1.0,
                result_cox['ct_true'] * (mean_var['pyogenes_var']['ct'])**(1/2) + mean_var['pyogenes_mean']['ct'],
                result_cox['ct_true'] * (mean_var['tuberculosis_var']['ct'])**(1/2) + mean_var['tuberculosis_mean']['ct'])

print(result_cox.head())

result_cox.to_csv(cox_result_name)