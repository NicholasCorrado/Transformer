import copy
from datetime import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable

# TODO: Put all this in EncoderDecoder
from attention import MultiHeadedAttention
from encoder_decoder import EncoderDecoder, Encoder, Decoder, Generator
from layers import PositionwiseFeedForward, Embeddings, EncoderLayer, DecoderLayer, LabelSmoothing
from noam_opt import NoamOpt
from postional_encoding import PositionalEncoding_v2
from utils import Batch, SimpleLossCompute


def make_model(src_vocab,
               tgt_vocab,
               N=6,
               d_model=512,
               d_ff=2048,
               h=8,
               dropout=0.1,
               encoding_mode='sinusoidal',
               combining_mode='add',
               max_wavelength=10_000.0):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding_v2(d_model,
                                     dropout,
                                     encoding_mode=encoding_mode,
                                     combining_mode=combining_mode,
                                     max_wavelength=max_wavelength)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        data = data.type(torch.int64)
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        # if i % 50 == 1:
        #     elapsed = time.time() - start
        #     print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
        #             (i, loss / batch.ntokens, tokens / elapsed))
        #     start = time.time()
        #     tokens = 0
    return total_loss / total_tokens


def train(model):
    # Train the simple copy task.
    criterion = LabelSmoothing(size=11, padding_idx=0, smoothing=0.0)
    # model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0.0001))  # lr=0.000001

    train_loss_avg_list = []
    val_loss_avg_list = []

    for epoch in range(20):
        model.train()
        train_loss_avg = run_epoch(data_gen(11, 30, 20), model,
                                   SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        val_loss_avg = run_epoch(data_gen(11, 30, 5), model,
                                 SimpleLossCompute(model.generator, criterion, None))
        val_loss_avg = np.float(val_loss_avg)

        print('Epoch: %i   Training Loss: %f   Validation Loss: %f' % (epoch, train_loss_avg, val_loss_avg))

        train_loss_avg_list.append(train_loss_avg)
        val_loss_avg_list.append(val_loss_avg)
    return train_loss_avg_list, val_loss_avg_list

