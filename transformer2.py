# -*- coding: utf-8 -*-
"""
Created on Thu May 27 23:51:06 2021

@author: bjorn
"""

from utils import *
import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
    """
    Model : AF/ normal classification
    Input: 10s, 300 Hz ECG signal + RRI for the same 10s 300 Hz ECG signal
    Output: Classification of input signal, either AF or Normal
    """
    def __init__(self, device, dim_val, dim_attn, input_size=1, signal_len=3000, num_class=2, n_encoder_layers=1, n_heads=1, rri_len=10):
        super(Transformer, self).__init__()
        self.device = device
        self.pos = PositionalEncoding(dim_val)
        self.signal_len = signal_len
        self.rri_len = rri_len
        self.dropout = torch.nn.Dropout(p=0.1)
        self.dropout_enc = torch.nn.Dropout(p=0.2)
        self.relu = torch.nn.ReLU()
        # Transformer Conv. layers
        self.conv1 = torch.nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=10, stride=3, padding=2)
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, stride=3, padding=0)
        self.conv3 = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=2, padding=0)
        self.conv4 = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=2, padding=0)
        self.relu_2_layer = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=4)
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=2)
        # self.conv1d_4_layer = torch.nn.Conv1d(in_channels=3, out_channels=10, kernel_size=3, bias=True)
        self.relu_4_layer = torch.nn.ReLU()
        # Transformer Encoder layers
        self.encs = []
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(self.device, dim_val, dim_attn, n_heads).cuda())
        # Transformer output Linear layers
        self.enc_input_fc = nn.Linear(input_size, dim_val)
        self.enc_fc1 = nn.Linear(128 * dim_val, 64)
        self.enc_fc2 = nn.Linear(64, 64)
        # RRI layers
        self.conv1_rri = torch.nn.Conv1d(in_channels=input_size, out_channels=60, kernel_size=3)
        self.conv2_rri = torch.nn.Conv1d(in_channels=60, out_channels=80, kernel_size=3) 
        self.maxpooling1_rri = torch.nn.MaxPool1d(kernel_size=2)
        # RRi Linear output layers
        self.flatten_layer = torch.nn.Flatten()
        self.dense1_rri = torch.nn.Linear(240, 64) 
        self.dense2_rri = torch.nn.Linear(64, 64) 
      
        # Linear output layer after concat.
        self.fc_out1 = torch.nn.Linear(64+64, 64)
        self.fc_out2 = torch.nn.Linear(64, 1) # if two classes problem is binary
        # if num_class==2:
        #   self.fc_out2 = torch.nn.Linear(64, 1) # if two classes problem is binary
        # else:
        #   self.fc_out2 = torch.nn.Linear(64, num_class) # if more than two classes problem is mulitclass

    def forward(self, x, x2): # x=ECG signal, x2=RRI
        # ECG signal
        # print('x shape before:', x.shape)
        x = x.view(-1,1,self.signal_len) # Resize to [1,1,1000] ([batch_size, input_channels, signal_length])
        # x = self.maxpool0(x)
        # print(x.shape)
        # Encoder layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        # print('shape after maxpool1:', x.shape)
        # x = self.dropout_layer(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        # print('shape after maxpool2:', x.shape)
        # CNN output is: [batch_size x hidden_size x seq_len]
        # x = x.view(x.size(0), x.size(2), x.size(1)) # with batch_first=True the input to LSTM should be [batch_size, seq_len, features].
        # x, (hidden_n, cell_n) = self.rnn1(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        # print('shape after maxpool3:', x.shape)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        # x = self.dropout(x)
        # print('shape after maxpool4:', x.shape)
        # Transformer Encoder
        e = self.encs[0](self.pos(self.enc_input_fc(x.to(self.device))))
        for enc in self.encs[1:]:
            e = enc(e)
        # Transformer Encoder output
        # print('Before transformer linear output', e.flatten(start_dim=1).shape)
        x = self.relu(self.enc_fc1(e.flatten(start_dim=1)))  
        # x = self.enc_fc2(self.dropout_enc(x)) 
        x = self.enc_fc2(x)

        # RRI conv. layers
        x2 = x2.view(-1,1,self.rri_len) # Resize to [1,1,100] ([batch_size, input_channels, rri_length])
        x2 = self.relu(self.conv1_rri(x2))
        x2 = self.relu(self.conv2_rri(x2))
        x2 = self.maxpooling1_rri(x2) 
        # RRI output layer
        x2 = self.flatten_layer(x2)
        # print('shape after RRI flatten:', x2.shape)
        x2 = self.dense2_rri(self.relu(self.dense1_rri(x2)))
        
        xc = torch.cat((x, x2), dim=1)
        # Linear output layer after concat.
        xc = self.flatten_layer(xc)
        # print('shape after flatten', x.shape)
        xc = self.relu(self.fc_out1(xc)) # hardcoded input dimension, try to change input dimension to be dynamic
        xc = self.dropout(xc)
        xc = self.fc_out2(xc) # hardcoded input dimension, try to change input dimension to be dynamic
        # x = self.dropout(x)
        # print(xc[0])
        return xc

# model = Transformer(signal_len=3000, num_class=2)
# model = model.to(device) 