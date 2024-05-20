import pandas as pd
import numpy as np
import argparse
from pandas.core.frame import DataFrame
from scipy.stats import zscore
import pdb

parser = argparse.ArgumentParser(description = 'parser for normalization')
parser.add_argument('--data_path', default = './data/210916/210916_train.csv', help = 'data path for normalization (default: ./data/train_210729+210820_drop.csv' )
parser.add_argument('--save_path', default = './data/210916/210916_train_cox', help = 'path for normalized data (default: ./data/normalized_210729+210820_drop.csv')
parser.add_argument('--save_mean_var_path', default = './data/210916/210916_cox_mean_var.csv', help = 'path for mean, var data (default: ./data/train_df_210910_mean_var.csv')
parser.add_argument('--drop', type = bool, default = False, help='True to remove nan ')
parser.add_argument('--z', type=bool, default = True, help = 'default=True, if test_data= False')
args = parser.parse_args()



class Normalization():
    def __init__(self, data_path):
        dataset = pd.read_csv(data_path)
        if args.drop == True:
            dataset = dataset.dropna(subset=['ct']).reset_index()
        if sum(dataset['ct'].isna()) !=0:
            dataset = dataset.fillna({'ct': 40.00})
        dataset['class'] = np.where((dataset['ct'] >= 40), 0, 1)
       

        py = dataset[dataset.species=='py']
        tb = dataset[dataset.species == 'tb']

        self.dataset = dataset
        self.py = py
        self.tb = tb
    
    def mean_var(self,mean_var_path, z):
        if args.z == True:
            py = self.py[['ct', 'RFU', 'DW_ct', 'DW_RFU']]
            tb = self.tb[['ct', 'RFU', 'DW_ct', 'DW_RFU']]

            py_mean, py_var = py.mean(axis = 0), py.var(axis = 0)
            tb_mean, tb_var = tb.mean(axis=0), tb.var(axis = 0)

            mean_var_data = pd.DataFrame({'py_mean':py_mean, 'py_var': py_var, 'tb_mean': tb_mean, 'tb_var':tb_var})

            mean_var_data.to_csv(mean_var_path)
   
   
    def Zscore(self, x, mean, var):
        return (x-mean)/var**0.5
    
    def get_item(self, save_path, z):
        features = ['ct', 'RFU', 'DW_ct', 'DW_RFU']
        py = self.py
        tb = self.tb
        datalist = [py, tb]
        
        if z == False:
            mean_var = pd.read_csv(args.save_mean_var_path)
            for i, data in enumerate(datalist):
                for f in features:
                    data[f'z_{f}'] = self.Zscore(data[f], mean_var[f'{data}_mean'][f], mean_var[f'{data}_var'][f])
                    del data[f]
                    if i == 1:
                        del self.dataset[f]
        
        else:
            for i, data in enumerate(datalist):
                for f in features:
                    data[f'z_{f}'] = zscore(data[f])
                    del data[f]
                    if i == 1:
                        del self.dataset[f]
           
            
        

        norm_data = pd.concat([self.py,self.tb], ignore_index=True)
        #to make normalized_ct >=0
        norm_data['z_ct'] += abs(norm_data['z_ct'].min())
        znorm_dataset = norm_data[['F_primer', 'R_primer', 'z_ct', 'z_RFU', 'z_DW_ct', 'z_DW_RFU', 'species', 'class']]
        znorm_dataset.columns = ['F_primer', 'R_primer', 'ct', 'RFU', 'dw_ct', 'dw_RFU', 'species', 'class']
      
        print(znorm_dataset.head())
        znorm_dataset.to_csv(save_path, index=False)
    
  
if args.drop == True:
    save_path = args.save_path+'_drop.csv'
else:
    save_path = args.save_path + '.csv'


data = Normalization(args.data_path)
data.mean_var(args.save_mean_var_path, args.z)
data.get_item(save_path, args.z)



