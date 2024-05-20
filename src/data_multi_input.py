import numpy as np
import torch
import torch.utils.data as data

import pdb


def merge_reg(targets_cla, targets_reg, targets_rank, species):
    lengths = [len(target) for target in targets_cla]
    target_merge_cla = torch.zeros(len(targets_cla), max(lengths)).float()
    target_merge_reg = torch.zeros(len(targets_reg), max(lengths)).float()
    target_merge_rank = torch.zeros(len(targets_rank), max(lengths)).float()
    species_merge = torch.zeros(len(targets_cla),2).float()
    for i, t in enumerate(targets_cla):
        end = lengths[i]
        target_merge_cla[i, :end] = t[:end]
    for i, t in enumerate(targets_reg):
        end = lengths[i]
        target_merge_reg[i, :end] = t[:end]
    for i, t in enumerate(targets_rank):
        end = lengths[i]
        target_merge_rank[i, :end] = t[:end]
    for i, s in enumerate(species):
        if s == 'py':
            species_merge[i][0] = 1
        else:
            species_merge[i][1] = 1
    return target_merge_cla, target_merge_reg, target_merge_rank, species_merge


#------------------------------------------------------------------------
# Data utils for MultiInputCNN
# author: sumin seo
#------------------------------------------------------------------------
class Dataset_FRP(data.Dataset):
    
    def __init__(self, data, key):
        
        self.data = data
        self.key = key
        self.num_total_seqs = len(self.data)
        self.max_seq_len = 0

        # rank_loss
        if 'rank' not in self.data:
            # generate ranks
            self.data['rank'] = self.data['ct'].rank(method = 'min', ascending = True, na_option = 'bottom')
        if 'class' not in self.data:
            self.data['class'] = np.where((self.data['ct'].isna() | self.data['ct'] >= 40), 0, 1)


    def __getitem__(self, index):
        #Fprimer:f/0, probe:b/1, Rprimer:r/2
        fprimer = self.data.iloc[index]['F_primer']
        #probe = self.data.iloc[index]['probe']
        rprimer = self.data.iloc[index]['R_primer']
        #prd_seq = self.data.iloc[index]['Product Sequence']
        species = self.data.iloc[index]['species']
        
        target_reg = self.data.iloc[index]['ct']  # 'ct'
        target_reg = torch.Tensor(np.array([target_reg]))

        if 'class' in self.data.iloc[index]:
            target_cla = self.data.iloc[index]['class']
        else:
            target_cla = torch.zeros(target_reg.size())
        target_cla = torch.Tensor(np.array([target_cla]))
        
        # rank_loss
        if 'rank' in self.data.iloc[index]:
            target_rank = self.data.iloc[index]['rank']
        else:
            target_rank = torch.zeros(target_reg.size())
        target_rank = torch.Tensor(np.array([target_rank]))

#         return (seq_f, seq_p, seq_r, seq_ps), target
        # special treat is needed for R primer (so didn't use numpy array)
        return fprimer, rprimer, target_cla, target_reg, target_rank, species

    def __len__(self):
        return self.num_total_seqs
    
    def set_max_seq_len(self, max_seq_len=None):
        if max_seq_len is None:
            self.max_seq_len = max([len(i) for i in self.data[self.key]])
        else:
            self.max_seq_len = max_seq_len

    
class CollateMultiCNN:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """
    def __init__(self, max_seq_len, key, word2index:dict, is_test=False):
        self.max_seq_len = max_seq_len
        self.word2id = word2index
        self.is_test = is_test
        self.key = key
        
    def collate_CNN_fn(self, data):

        # one hot -> list (to cover multiple columns)
        # DO NOT SORT or SHUFFLE
        fprimer, rprimer, targets_cla, targets_reg, targets_rank, species = zip(*data)
        fprimer = self.one_hot_encode(fprimer, self.max_seq_len, self.word2id)
        #probe = self.one_hot_encode(probe, self.max_seq_len, self.word2id)
        rprimer = self.one_hot_encode(rprimer, self.max_seq_len, self.word2id, key='R primer')
            
        tars_cla, tars_reg, tars_rank, species = merge_reg(targets_cla, targets_reg, targets_rank, species)
        if self.is_test:
            return (fprimer, rprimer), tars_cla, tars_reg, tars_rank, species
        else:
            return (fprimer, rprimer), tars_cla, tars_reg, tars_rank, species
            
    def one_hot_encode(self, seqs, max_seq_len, word2id, key=None):
        # use special base dict for R/F primer and probe
        base_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'R': [0,2], 'Y': [1,3], 'M': [0,3], 'K': [1,2]}
       
        if key == 'R primer':
            seqs = list(seqs)
            for i in list(range(len(seqs))):
                seqs[i] = self.comp(seqs[i])
            seqs = tuple(seqs)

        lengths = [len(seq) for seq in seqs]
        
        #len(seqs)*len(self.word2id)*self.max_seq_len
        onehot = torch.zeros(len(seqs), len(word2id), max_seq_len).float()
        
        for i, s in enumerate(seqs):
            for j, base in enumerate(s):
                try:
                    index = i, base_dict[base],j
                except:
                    print(base)
                onehot[index] = 1
        return onehot
    
    def comp(self, seq):
        compDict = {"A":"T", "G":"C", "T":"A", "C":"G", "Y":"R", "M":"K", "R":"Y", "K":"M"}
        retList = []
        for ele in seq:
            if ele not in compDict:
                continue
            retList.append(compDict[ele])
        
        return "".join(reversed(retList))    
    
    def __call__(self, batch):
        return self.collate_CNN_fn(batch)
    

def get_loader_CNN(dataset, subsampler, batch_size, key, word2dict:dict, max_seq_len=40, is_test=False):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              sampler=subsampler,
                                              collate_fn=CollateMultiCNN(max_seq_len, key, word2dict, is_test))
    
    return data_loader
    
    
def get_loader_CNN_infer(dataset, batch_size, key, word2dict:dict, max_seq_len=40, is_test=True):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=CollateMultiCNN(max_seq_len, key, word2dict, is_test))
    
    return data_loader

