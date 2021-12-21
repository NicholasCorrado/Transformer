import math

import torch
from torch import nn
from torch.autograd import Variable
from global_vars import DEVICE 

class PositionalEncoding_v2(nn.Module):
    "Implement the PE function."
    def __init__(self,
                 d_model,
                 dropout,
                 encoding_mode = 'sinusoidal',
                 combining_mode = 'add',
                 max_len=5000,
                 max_wavelength=10_000.0):
        super(PositionalEncoding_v2, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.encoding_mode = encoding_mode
        self.combining_mode = combining_mode
        self.de = nn.Linear(d_model * 2, d_model)
        self.at = nn.Tanh()
        if encoding_mode == 'sinusoidal':
            # Compute the positional encodings once in log space.
            self.pe = torch.zeros(max_len, d_model).to(DEVICE)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) *
                                 -(math.log(max_wavelength) / d_model))
            self.pe[:, 0::2] = torch.sin(position * div_term)
            self.pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = self.pe.unsqueeze(0)
        elif encoding_mode == 'learnable':
            self.pe = nn.Parameter(torch.zeros(max_len, d_model).unsqueeze(0),
                              requires_grad=True)
        else:
            print(encoding_mode)
            raise NotImplementedError()
    def forward(self, x):
        if self.combining_mode == 'add':
            if self.encoding_mode == 'sinusoidal':
                x = x + Variable(self.pe[:, :x.size(1)],
                                 requires_grad=False)
                return self.dropout(x)
            else:
                x = x + self.pe[:, :x.size(1)]
                return self.dropout(x)
        elif self.combining_mode == 'concat':
            if self.encoding_mode == 'sinusoidal':
                x = torch.cat([x,Variable(self.pe[:, :x.size(1)].repeat(x.size(0),1,1),
                                 requires_grad=False)],axis=2)
                x = self.at(self.de(x))
                return self.dropout(x)
            else:
                x = torch.cat([x,self.pe[:, :x.size(1)].repeat(x.size(0),1,1)
                              ],axis=2)
                x = self.at(self.de(x))
                return self.dropout(x)
        else:
            print(encoding_mode)
            raise NotImplementedError()