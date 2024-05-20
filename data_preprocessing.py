import pandas as pd
import argparse

#parser
parser = argparse.ArgumentParser(description = 'Parser for preprocessing.')
parser.add_argument('--data_path',default = './data/210820_TB_raw_data.xlsx',
                    help = 'path for data (defalt: ./data/210820_TB_raw_data.xlsx' )
parser.add_argument('--save_path', default = './data/210820_TB.csv',
                    help = 'save path for preprocessed data (default: ./data/210820_TB.csv')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = pd.read_excel(args.data_path)


#find interest column
find_list = ['primer', 'sequence', 'ct', 'RFU', 'ct', 'RFU']
column_list = ['primer', 'primer_sequence', 'ct', 'RFU', 'DW_ct', 'DW_RFU']
new_column_list = [0]*len(dataset.columns)

for i in range(len(dataset.columns)):
    for j in range(len(find_list)):
        if any(dataset[dataset.columns[i]] == find_list[j]):
            new_column_list[i] = column_list[j]
            find_list[j] =0
            break 


#dataset slicing
new_index =[]
new_column = []
for index, column in enumerate(new_column_list):
    if column != 0:
        new_index.append(index)
        new_column.append(column)

dataset = dataset.iloc[3:, new_index]
dataset = dataset.reset_index(drop= True)
dataset.columns = new_column




#separate F primer and R primer
F_primer = []
R_primer = []
R_primer_index = []

for i in range(len(dataset.index)):
    if dataset.primer[i] == 'R primer':
        R_primer.append(dataset.primer_sequence[i])
        R_primer_index.append(i)


dataset.drop(R_primer_index, inplace = True)
dataset.rename(columns= {'primer_sequence' : 'F_primer'}, inplace = True)
dataset['R_primer'] = R_primer
del dataset['primer']


#save dataset
dataset = dataset[['F_primer', 'R_primer', 'ct', 'RFU', 'DW_ct', 'DW_RFU']]
dataset = dataset.reset_index(drop = True)
dataset.index = dataset.index + 1

dataset.to_csv(args.save_path, na_rep = "NaN")

print(dataset.head())