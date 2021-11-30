import os.path

import torch

from plot import visualize_all_curves

import copy
import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable

from attention import MultiHeadedAttention
from encoder_decoder import EncoderDecoder, Encoder, Decoder, Generator
from layers import PositionwiseFeedForward, Embeddings, EncoderLayer, DecoderLayer, LabelSmoothing
from noam_opt import NoamOpt
from postional_encoding import PositionalEncoding_v2
from utils import Batch, SimpleLossCompute
from layers import PricePreprocessor

# vocab size
V = 10

def make_model(src_vocab,
               tgt_vocab,
               N=6, # num encoder layers
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

def make_prediction_model(
               src_vocab,
               tgt_vocab,
               N=6, # num encoder layers
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

def data_gen_2(batches):
    for i in range(len(batches)):
        yield batches[i]

def batch_data():

    energy_data, energy_data_col_names = get_stock_data(file_path="data/energydata.csv")
    # energy_data = energy_data[:1000]

    pp = PricePreprocessor(stock_prices=energy_data, vocab_size=V)
    sentences, target_words = pp.get_rolling_window_sentences(window_length=50)

    perm = np.random.permutation(len(sentences))[:100]
    sentences = sentences[perm]
    target_words = target_words[perm]
    print(np.min(sentences), np.max(sentences))

    n = len(sentences)
    batch_size = 10

    n_train = int(n*0.80)
    n_test = n-n_train
    n_train_batches = math.ceil(n_train/batch_size)
    n_test_batches = math.ceil((n_test)/batch_size)

    sentences = torch.from_numpy(sentences).type(torch.int64)
    target_words = torch.from_numpy(target_words).type(torch.int64)

    train_batches = []
    test_batches = []

    for i in range(n_train_batches):
        start = i*batch_size
        end = (i+1)*batch_size
        src = sentences[start:end]
        trg = target_words[start:end]
        train_batches.append(Batch(src, trg))
    for i in range(n_test_batches):
        start = n_train+i*batch_size
        end = n_train+(i+1)*batch_size
        src = sentences[start:end]
        trg = target_words[start:end]
        test_batches.append(Batch(src, trg))

    return train_batches, test_batches

def get_stock_data(file_path):
    # TODO: need to properly read date column

    # Use -1 for missing values. Since values are stored in a numpy array,
    # filling_values must be set to a number (i.e. not None)
    data_energy = np.genfromtxt(fname=file_path,
                                delimiter=',',
                                skip_header=True,
                                filling_values=-1,
                                usecols=(1,))

    # Read first line to get column names
    file = open(file_path)
    col_names = file.readline().split(',')

    return data_energy, col_names
    #data_energy = np.loadtxt('./data/techdata.csv/')

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        print('batch num:', i)
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
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.1)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0.0001))  # lr=0.000001

    train_loss_avg_list = []
    val_loss_avg_list = []

    train_batches, test_batches = batch_data()

    for epoch in range(10):
        model.train() # set weights to training mode
        print('Training epoch...')
        train_loss_avg = run_epoch(data_gen_2(train_batches), model,
                                   SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print('Validation epoch...')
        val_loss_avg = run_epoch(data_gen_2(test_batches), model,
                                 SimpleLossCompute(model.generator, criterion, None))

        val_loss_avg = float(val_loss_avg)
        print('Epoch: %i   Training Loss: %f   Validation Loss: %f' % (epoch, train_loss_avg, val_loss_avg))

        train_loss_avg_list.append(train_loss_avg)
        val_loss_avg_list.append(val_loss_avg)
    return train_loss_avg_list, val_loss_avg_list

def save_model(encoder_decoder, path):
    torch.save(encoder_decoder, path)

if __name__ == "__main__":

    print('Creating models...')

    sa_model = make_model(V, V, N=2, encoding_mode='sinusoidal', combining_mode='add', max_wavelength=10000)
    la_model = make_model(V, V, N=2, encoding_mode='learnable', combining_mode='add', max_wavelength=10000)
    sc_model = make_model(V, V, N=2, encoding_mode='sinusoidal', combining_mode='concat', max_wavelength=10000)
    lc_model = make_model(V, V, N=2, encoding_mode='learnable', combining_mode='concat', max_wavelength=10000)

    print('Starting Training...')
    sc_train_loss, sc_val_loss = train(sc_model)
    lc_train_loss, lc_val_loss = train(lc_model)
    sa_train_loss, sa_val_loss = train(sa_model)
    la_train_loss, la_val_loss = train(la_model)

    np.save('sc_train_loss', np.array(sc_train_loss))
    np.save('lc_train_loss', np.array(lc_train_loss))
    np.save('sa_train_loss', np.array(sa_train_loss))
    np.save('la_train_loss', np.array(la_train_loss))

    np.save('sc_val_loss', np.array(sc_val_loss))
    np.save('lc_val_loss', np.array(lc_val_loss))
    np.save('sa_val_loss', np.array(sa_val_loss))
    np.save('la_val_loss', np.array(la_val_loss))

    #save_model(sc_model, 'results/sc_model_trained.pt')
    # save_model(lc_model, 'results/lc_model_trained.pt')
    # save_model(sa_model, 'results/sa_model_trained.pt')
    # save_model(la_model, 'results/la_model_trained.pt')

    visualize_all_curves(sa_train_loss, sa_val_loss, sc_train_loss, sc_val_loss, "sinusoidal addition/concat")
    visualize_all_curves(la_train_loss, la_val_loss, lc_train_loss, lc_val_loss, "learnable addition/concat")

    # sa_model = make_model(V, V, N=2, encoding_mode='sinusoidal', combining_mode='add')
    # la_model = make_model(V, V, N=2, encoding_mode='learnable', combining_mode='add')
    # sc_model = make_model(V, V, N=2, encoding_mode='sinusoidal', combining_mode='concat')
    # lc_model = make_model(V, V, N=2, encoding_mode='learnable', combining_mode='concat')