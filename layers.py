import math

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils import clones
from global_vars import DEVICE 

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # add extra layer here.
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PricePreprocessor:
    def __init__(self, stock_prices_1, stock_prices_2=None, vocab_size=1000):
        self.log_prices_1 = np.log(stock_prices_1)
        if np.any(np.isnan(self.log_prices_1)):
            print('WARNING: log_prices conains NaN. An entry from the data is probably missing and set to -1.')
            print('Truncating data at first NaN...')
            nan_index = np.where(np.isnan(self.log_prices))[0][0]
            self.log_prices_1 = self.log_prices[:nan_index]
        self.max_log_price = np.max(self.log_prices_1)
        self.min_log_price = np.min(self.log_prices_1)
        self.vocab_size = vocab_size

        # bins[i] = right endpoint of bin
        # e.g. bins = [0.5, 1] for bins [0, 0.5], [0.5, 1]
        self.bins = np.empty(vocab_size)
        self.bin_width = (self.max_log_price - self.min_log_price) / self.vocab_size
        for i in range(vocab_size):
            self.bins[i] = self.min_log_price + self.bin_width*(i+1)


        self.bins[-1] = np.inf
        self.n = len(self.log_prices_1)
        self.word_indices_1 = self.map_log_prices_to_word_index(self.log_prices_1)        

    def get_rolling_window_sentences(self, window_length=50):

        num_sentences = len(self.word_indices_1) - 2*window_length + 1

        sentences_1 = [self.word_indices_1[i:i+window_length] for i in range(num_sentences)]
        sentences_2 = [self.word_indices_1[i+window_length:i+2*window_length] for i in range(num_sentences)]

        sentences_1 = np.array(sentences_1)
        sentences_2 = np.array(sentences_2)
        
        return sentences_1, sentences_2

    def map_log_prices_to_word_index(self, input_sentence):
        '''
        input_prices: 1D array of dim (sentence_length=50) containing log prices in batch_dim sentences
        output: 1D array of dim (sentence_length=50) containing indices corresponding to the original log prices
        '''
        output = np.zeros(len(input_sentence))
        for i in range(len(input_sentence)):
            if len(np.where(input_sentence[i] <= self.bins)[0]) == 0:
                stop = 0
            output[i] = np.where(input_sentence[i] <= self.bins)[0][0]
        return output

    
    
class PricePreprocessorTranslation:
    def __init__(self, stock_prices_1, stock_prices_2=None, vocab_size=1000):
        self.log_prices_1 = np.log(stock_prices_1)
        self.log_prices_2 = np.log(stock_prices_2)
        
        if np.any(np.isnan(self.log_prices_1)):
            print('WARNING: log_prices conains NaN. An entry from the data is probably missing and set to -1.')
            print('Truncating data at first NaN...')
            nan_index = np.where(np.isnan(self.log_prices))[0][0]
            self.log_prices_1 = self.log_prices[:nan_index]
        max_1 = np.max(self.log_prices_1)
        max_2 = np.max(self.log_prices_2)
        print(max_1, max_2)
        self.max_log_price = max(max_1, max_2)
        min_1 = np.min(self.log_prices_1)
        min_2 = np.min(self.log_prices_2)
        self.min_log_price = min(min_1, min_2)    
        
        self.vocab_size = vocab_size

        # bins[i] = right endpoint of bin
        # e.g. bins = [0.5, 1] for bins [0, 0.5], [0.5, 1]
        self.bins = np.empty(vocab_size)
        self.bin_width = (self.max_log_price - self.min_log_price) / self.vocab_size
        for i in range(vocab_size):
            self.bins[i] = self.min_log_price + self.bin_width*(i+1)

        self.bins[-1] = np.inf
        self.n = len(self.log_prices_1)
        self.word_indices_1 = self.map_log_prices_to_word_index(self.log_prices_1)  
        self.word_indices_2 = self.map_log_prices_to_word_index(self.log_prices_2) 
        

    def get_rolling_window_sentences(self, window_length=50):

        num_sentences = len(self.word_indices_1) - window_length + 1

        sentences_1 = [self.word_indices_1[i:i+window_length] for i in range(num_sentences)]
        sentences_2 = [self.word_indices_2[i:i+window_length] for i in range(num_sentences)]

        sentences_1 = np.array(sentences_1)
        sentences_2 = np.array(sentences_2)
        return sentences_1, sentences_2

    def map_log_prices_to_word_index(self, input_sentence):
        '''
        input_prices: 1D array of dim (sentence_length=50) containing log prices in batch_dim sentences
        output: 1D array of dim (sentence_length=50) containing indices corresponding to the original log prices
        '''
        output = np.zeros(len(input_sentence))
        for i in range(len(input_sentence)):
            if len(np.where(input_sentence[i] <= self.bins)[0]) == 0:
                stop = 0
            output[i] = np.where(input_sentence[i] <= self.bins)[0][0]
        return output
    
if __name__ == "__main__":
    prices = np.arange(10)

    p = PricePreprocessor(prices, vocab_size=5)
    sentences = p.get_rolling_window_sentences(window_length=10)

    print(sentences)

    indices = p.map_log_prices_to_word_index(sentences[0])
    print(indices)
