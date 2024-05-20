import torch
import torch.nn as nn
import torch.nn.functional as F

import random


PAD_token = 0
SOS_token = 1  # 0은 패딩
EOS_token = 2


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidn_dim, device, dropout=0.5):
        super().__init__()

        self.device = device

        self.input_dim = input_dim
        self.hidn_dim = hidn_dim
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)  
        self.lstm = nn.LSTM(emb_dim, hidn_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, prev_hidden, prev_cell):
        """
        inputs: (batch_size, one_hot_encode, hidn_dim)
        prev_hidden: (batch_size, hidn_dim, hidn_dim)
        embedded: (batch_size, seq_len, hidn_size)
        output: (batch_size, seq_len, hidn_size)
        hidden: (batch_size, seq_len, hidn_size)
        cell: (batch_size, seq_len, hidn_size)
        """
        inputs_gpu = inputs.to(self.device)
        embedded = self.embedding(inputs_gpu)
        output, (hidden, cell) = self.lstm(embedded, (prev_hidden, prev_cell))
        output = self.dropout(output)
        
        return output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidn_dim, device):
        super().__init__()
        
        self.device = device
        
        self.output_dim = output_dim
        self.hidn_dim = hidn_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidn_dim, batch_first=True)
        self.out = nn.Linear(hidn_dim, output_dim)
    
    def forward(self, inputs, prev_hidden, prev_cell):
        """
        * use batch_first = True
        inputs: (batch_size, 1)
        embedded: (batch_size, 1, hidn_size)
        output: (batch_size, 1, hidn_size)
        hidden: (batch_size, 1, hidn_size)
        cell: (batch_size, 1, hidn_size)
        """
        inputs = inputs.unsqueeze(1)
        inputs_gpu = inputs.to(self.device)
        embedded = self.embedding(inputs_gpu)
        output = F.relu(embedded)  # nn.ReLU랑 차이점?
        output, (hidden, cell) = self.lstm(output, (prev_hidden, prev_cell))

        output = self.out(output)

        output = output.squeeze(1)

        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, inputs, targets, teacher_forcing_ratio = 0.5):   
        batch_size = targets.size(0)  # batch_first
        input_len = inputs.size(1)
        target_len = targets.size(1)
        enc_hidn_dim = self.encoder.hidn_dim
        tar_vocab_size = self.decoder.output_dim

        enc_hidn = torch.zeros(1, batch_size, enc_hidn_dim, device=self.device)
        enc_cell = torch.zeros(1, batch_size, enc_hidn_dim, device=self.device)
        dec_outputs = torch.zeros(batch_size, target_len, tar_vocab_size, device=self.device)
        predictions = torch.zeros(batch_size, target_len, device=self.device)  # show results
        
        # Encode
        enc_outputs, enc_hidn, enc_cell = self.encoder.forward(inputs, enc_hidn, enc_cell)


        # Decode
        dec_input = targets[:, 0]  # batch_first
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        predictions[:, 0] = dec_input

        # sequence 길이 만큼 iteration
        for i in range(1, target_len):
            dec_output, dec_hidn, dec_cell = self.decoder.forward(dec_input, enc_hidn, enc_cell)

            dec_outputs[:, i] = dec_output
            target = targets[:, i]

            if use_teacher_forcing:
                dec_input = target
            else:
                dec_input = dec_output.argmax(1)  # (batch_size,)
            
            predictions[:, i] = dec_input
            
        return dec_outputs, predictions
    
    
class Regressor(nn.Module):
    def __init__(self, input_dim, hidn_dim, device, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
#         self.hidn_dim = hidn_dim
        self.device = device
        self.dense_hidn = nn.Linear(input_dim, hidn_dim)
        self.linear = nn.Linear(hidn_dim, 16)
        self.dense_out = nn.Linear(16, 1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, enc_output):
        """ 
        inputs: sequence latent representation
            - (batch_size, seq_len, hidn_size)
        targets: RFU, CT
            - (batch_size, 1)
        """
        # 미리 학습한 encoder를 통과시킨 output 사용
#         print(prev_hidn) # (1, batch, hid)
#         print(prev_cell) # (1, batch, hid)
#         print(inputs) # (batch, seq, hid)
        
        # encoder output -> regression
        pool = nn.MaxPool1d(enc_output.shape[1])
        output = pool(enc_output.permute(0,2,1)).permute(0,2,1)
        output = self.flatten(output)
#         print('output1 size : %s' % str(output.size()))
#         print(output)
        
        output = self.dropout(self.dense_hidn(output))
        output = self.dropout(self.linear(output))
        output = self.dense_out(output)
#         print('output2 size : %s' % str(output.size()))
#         print(output)
        
        return output
    
    
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
        self.dense_hidn = nn.Linear(vocab_size * (input_dim//5), hidn_dim)  # input_dim//input_dim=1
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
        #print('inputs size : %s' % str(inputs.size()))
        # input (batch_size, vocab_size, input_dim)
        output = F.relu(self.conv(inputs))
        # conv output (batch_size, vocab_size, input_dim)
        #print('output conv size : %s' % str(output.size()))

        # sequence axis로 max pooling
#         output = self.maxpool(output.permute(0,2,1)).permute(0,2,1)
        output = self.maxpool(output)
        # maxpool output (batch_size, vocab_size, input_dim//kernel_size)
        #print('output maxpool size : %s' % str(output.size()))
#         print(output)
        
        output = self.flatten(output)
        # flatten output (batch_size, vocab_size * (input_dim//kernel_size))
        #print('output flatten size : %s' % str(output.size()))
        
        output = self.dropout(self.dense_hidn(output))
        # dense_hidn output (batch_size, hidn_dim)
        #print('output dense_hidn size : %s' % str(output.size()))

#         output = self.dropout(self.linear(output))
        output = self.dense_out(output)
        # dense_out output (batch_size, 1)
        #print('output dense_out size : %s' % str(output.size()))
#         print(output)
        
        return output