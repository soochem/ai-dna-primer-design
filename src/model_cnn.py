import torch
import torch.nn as nn
import torch.nn.functional as F

import random


#------------------------------------------------------------------------
# For CNN
# author: sumin seo
#------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, input_dim, vocab_size, hidn_dim, device, kernel_size=3, dropout=0.3):
        super().__init__()
        self.device = device
        self.input_dim = input_dim    # max_seq_len
        self.vocab_size = vocab_size
        self.hidn_dim = hidn_dim      # hidn_dim for dense layer
        
        # 8 input channel, 8 output channels, 3x3 square convolution
        # same padding
        self.conv = nn.Conv1d(vocab_size, vocab_size, kernel_size, padding=(kernel_size-1)//2)  # 차원 늘리는 건 좀 더 고민 필요
        self.maxpool = nn.MaxPool1d(5) #input_dim
        self.dense_hidn = nn.Linear(vocab_size * (input_dim//4), hidn_dim)  # input_dim//input_dim=1
#         self.linear = nn.Linear(hidn_dim, 21)
        self.dense_out = nn.Linear(hidn_dim, 1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        """ 
        inputs: sequence one-hot encoding
            - (batch_size, vocab_size, max_seq_len)
        targets: RFU, CT
            - (batch_size, 1)
        """
#         print('inputs size : %s' % str(inputs.size()))
        # input (batch_size, vocab_size, input_dim)
        output = F.relu(self.conv(inputs))
        # conv output (batch_size, vocab_size, input_dim)
#         print('output conv size : %s' % str(output.size()))

        # sequence axis로 max pooling
#         output = self.maxpool(output.permute(0,2,1)).permute(0,2,1)
        output = self.maxpool(output)
        # maxpool output (batch_size, vocab_size, input_dim//kernel_size)
#         print('output maxpool size : %s' % str(output.size()))
#         print(output)
        
        output = self.flatten(output)
        # flatten output (batch_size, vocab_size * (input_dim//kernel_size))
#         print('output flatten size : %s' % str(output.size()))
        
        output = self.dropout(self.dense_hidn(output))
        # dense_hidn output (batch_size, hidn_dim)
#         print('output dense_hidn size : %s' % str(output.size()))

#         output = self.dropout(self.linear(output))
        output = self.dense_out(output)
        # dense_out output (batch_size, 1)
#         print('output dense_out size : %s' % str(output.size()))
#         print(output)
        
        return output


class MultiInputCNN(nn.Module):
    def __init__(self, input_dim, vocab_size, hidn_dim, device, kernel_size=3, dropout=0.3, return_hidn_state=False):
        super().__init__()
        self.device = device
        self.input_dim = input_dim    # max_seq_len
        self.vocab_size = vocab_size
        self.hidn_dim = hidn_dim      # hidn_dim for dense layer
        self.pool_size = 4            # no input
        
        kernel_1 = 3
        kernel_2 = 5
        kernel_3 = 7
        # same padding
        self.conv_f_1 = nn.Conv1d(vocab_size, vocab_size, kernel_1, padding=(kernel_1-1)//2)
        self.conv_f_2 = nn.Conv1d(vocab_size, vocab_size, kernel_2, padding=(kernel_2-1)//2)
        self.conv_f_3 = nn.Conv1d(vocab_size, vocab_size, kernel_3, padding=(kernel_3-1)//2)
        self.conv_p_1 = nn.Conv1d(vocab_size, vocab_size, kernel_1, padding=(kernel_1-1)//2)
        self.conv_p_2 = nn.Conv1d(vocab_size, vocab_size, kernel_2, padding=(kernel_2-1)//2)
        self.conv_p_3 = nn.Conv1d(vocab_size, vocab_size, kernel_3, padding=(kernel_3-1)//2)
        self.conv_r_1 = nn.Conv1d(vocab_size, vocab_size, kernel_1, padding=(kernel_1-1)//2)
        self.conv_r_2 = nn.Conv1d(vocab_size, vocab_size, kernel_2, padding=(kernel_2-1)//2)
        self.conv_r_3 = nn.Conv1d(vocab_size, vocab_size, kernel_3, padding=(kernel_3-1)//2)
        
        self.maxpool_f = nn.MaxPool1d(self.pool_size)
        self.maxpool_p = nn.MaxPool1d(self.pool_size)
        self.maxpool_r = nn.MaxPool1d(self.pool_size)
        
        self.dense_hidn = nn.Linear(3 * vocab_size*2 * input_dim, hidn_dim)
        self.dense_hidn_species = nn.Linear(3 * vocab_size*2 * input_dim +2, hidn_dim)  # input_dim//input_dim=1
        #input_dim * self.pool_size for maxpooling
        #self.linear = nn.Linear(hidn_dim, 32)
        self.dense_out1 = nn.Linear(hidn_dim, 1)
        self.dense_out2 = nn.Linear(64, 1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)

        # for cox
        self.return_hidn_state = return_hidn_state
        
    def forward(self, inputs, species=None):
        """ 
        inputs: sequence one-hot encoding
            - (batch_size, vocab_size, max_seq_len)
        targets: RFU, CT
            - (batch_size, 1)
        """
        
#         print(type(inputs))
        fprimer = inputs[0].to(self.device)
        #probe = inputs[1].to(self.device)
        rprimer = inputs[1].to(self.device)
        # Model 1
        # print('fprimer input size : %s' % str(fprimer.size()))
        # input (batch_size, vocab_size, input_dim)
        fprimer_output_1 = F.relu(self.conv_f_1(fprimer))
        fprimer_output_2 = F.relu(self.conv_f_2(fprimer))
        fprimer_output_3 = F.relu(self.conv_f_3(fprimer))
        # fprimer_output_2 size (batch_size, vocab_size, input_dim)
        # Concat along vocab embedding axis
        fprimer_output = torch.cat((fprimer_output_1, fprimer_output_2, fprimer_output_3), 1)
        # print('frimer cat size : %s' % str(fprimer_output.size()))
        # fprimer_output size (batch_size, vocab_size*2, input_dim)
        #fprimer_output = self.maxpool_f(fprimer_output)
        # fprimer_output size (batch_size, vocab_size*2, input_dim//pool_size)
        fprimer_output = self.flatten(fprimer_output)
        # fprimer_output size (batch_size*vocab_size*2*input_dim//pool_size)
        # print('frimer flatten size : %s' % str(fprimer_output.size()))
        """
        # Model 2
        probe_output_1 = F.relu(self.conv_p_1(probe))
        probe_output_2 = F.relu(self.conv_p_2(probe))
        probe_output_3 = F.relu(self.conv_p_3(probe))
        probe_output = torch.cat((probe_output_1, probe_output_2, probe_output_3), 1)
        probe_output = self.maxpool_p(probe_output)
        probe_output = self.flatten(probe_output)
        """
        # Model 3
        rprimer_output_1 = F.relu(self.conv_r_1(rprimer))
        rprimer_output_2 = F.relu(self.conv_r_2(rprimer))
        rprimer_output_3 = F.relu(self.conv_r_3(rprimer))
        rprimer_output = torch.cat((rprimer_output_1, rprimer_output_2, rprimer_output_3), 1)
        #rprimer_output = self.maxpool_r(rprimer_output)
        rprimer_output = self.flatten(rprimer_output)
        
        # an output (batch_size, vocab_size*2, input_dim//pool_size)
        output = torch.cat((fprimer_output, rprimer_output), 1)
        if species is not None:
            species = species.to(self.device)
            output = torch.cat((output, species),1)
            #print(output.shape)
            output = self.dropout(self.dense_hidn_species(output))        
        #print('output cat size : %s' % str(output.size()))
        # output size (3 * batch_size*vocab_size*2*input_dim//pool_size)
        #output = self.dense_hidn(output)
        else:
            print(output.shape)
            output = self.dropout(self.dense_hidn(output))
        # dense_hidn output (batch_size, hidn_dim)
#         print('output dense_hidn size : %s' % str(output.size()))
        #output = self.dropout(self.linear(output))

        if self.return_hidn_state:
            return output

        output = self.dense_out1(output)
        #output = self.dense_out2(output)
        # dense_out output (batch_size, 1)
#         print('output dense_out size : %s' % str(output.size()))
#         print(output)
        
        return output