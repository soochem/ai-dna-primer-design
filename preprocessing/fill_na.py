import pandas as pd
import numpy as np
import csv
import pdb
from os import path

train_df = pd.read_csv('../data/211020_train_2.csv')
test_df = pd.read_csv('../data/211020_test_2.csv')

print(train_df.head())
print(test_df.head())

train_df['ct'] = np.where(train_df['ct'].isna(), 40, train_df['ct'])
test_df['ct'] = np.where(test_df['ct'].isna(), 40, test_df['ct'])

print(train_df.head())
print(test_df.head())

pdb.set_trace()

train_df.to_csv('./data/211020_train_fillna.csv', index=0)
test_df.to_csv('./data/211020_test_fillna.csv', index=0)