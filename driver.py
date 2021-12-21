import os.path
import time

import torch
from torch.utils.data import TensorDataset, DataLoader

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
from layers import PricePreprocessor, PricePreprocessorTranslation
from global_vars import DEVICE

import argparse

# Subset of S&P 500 constituents for which we have data for all dates from 1/3/2007 - 10/8/2021
SP500_CONSTITUENTS_ENERGY = ['APA', 'BKR', 'COP', 'CTRA', 'CVX', 'DVN', 'EOG', 'HAL', 'HES', 'MRO', 'OXY', 'PXD', 'SLB',
                             'VLO', 'XOM']

SP500_CONSTITUENTS_TECH = ['AAPL' 'ACN' 'ADBE' 'ADI' 'ADSK' 'AMD' 'ANSS' 'ATVI' 'CDNS' 'CRM' 'CSCO'
                           'CTSH' 'CTXS' 'DXC' 'EA' 'EFX' 'FFIV' 'FISV' 'GOOG' 'GOOGL' 'HPQ' 'IBM'
                           'INTC' 'INTU' 'IPG' 'JKHY' 'JNPR' 'LRCX' 'MCHP' 'MCO' 'MPWR' 'MSFT' 'MSI'
                           'MU' 'NLOK' 'NTAP' 'NVDA' 'OMC' 'ORCL' 'PTC' 'QCOM' 'RHI' 'ROL' 'SNPS'
                           'SPGI' 'STX' 'SWKS' 'TTWO' 'TXN' 'TYL' 'URI' 'VRSN' 'WDC' 'XLNX' 'ZBRA']


def make_model(src_vocab,
               tgt_vocab,
               N=6,  # num encoder layers
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
        N=6,  # num encoder layers
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


def copy_task_data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        data = data.type(torch.int64)
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


def yield_batch(dl_src, dl_trg):
    for (src, trg) in zip(dl_src, dl_trg):
        src = src[0].to(DEVICE)
        trg = trg[0].to(DEVICE)
        yield Batch(src, trg)


def batch_data_price_prediction(stock_data, vocab_size, batch_size=64, window_length=50):
    pp = PricePreprocessor(stock_prices_1=stock_data, vocab_size=vocab_size)
    sentences, target_words = pp.get_rolling_window_sentences(window_length=window_length)

    print(np.min(sentences), np.max(sentences))

    n = len(sentences)
    n_train = int(n * 0.80)
    n_test = n - n_train
    n_train_batches = math.ceil(n_train / batch_size)
    n_test_batches = math.ceil((n_test) / batch_size)

    perm = np.random.permutation(n_train)
    sentences[np.arange(n_train)] = sentences[perm]
    target_words[np.arange(n_train)] = target_words[perm]

    sentences = torch.from_numpy(sentences).type(torch.int64)
    target_words = torch.from_numpy(target_words).type(torch.int64)

    train_dl_src = DataLoader(TensorDataset(sentences[:n_train]), batch_size=batch_size)
    train_dl_trg = DataLoader(TensorDataset(target_words[:n_train]), batch_size=batch_size)

    val_dl_src = DataLoader(TensorDataset(sentences[n_train:]), batch_size=batch_size)
    val_dl_trg = DataLoader(TensorDataset(target_words[n_train:]), batch_size=batch_size)

    return (train_dl_src, train_dl_trg), (val_dl_src, val_dl_trg)


def batch_data_price_translation(stock_data_1, stock_data_2, vocab_size, batch_size=64, window_length=50):
    pp = PricePreprocessorTranslation(stock_prices_1=stock_data_1, stock_prices_2=stock_data_2, vocab_size=vocab_size)
    sentences, target_words = pp.get_rolling_window_sentences(window_length=window_length)

    print(np.min(sentences), np.max(sentences))

    n = len(sentences)
    n_train = int(n * 0.80)
    n_test = n - n_train
    n_train_batches = math.ceil(n_train / batch_size)
    n_test_batches = math.ceil((n_test) / batch_size)

    perm = np.random.permutation(n_train)
    sentences[np.arange(n_train)] = sentences[perm]
    target_words[np.arange(n_train)] = target_words[perm]

    sentences = torch.from_numpy(sentences).type(torch.int64)
    target_words = torch.from_numpy(target_words).type(torch.int64)

    train_dl_src = DataLoader(TensorDataset(sentences[:n_train]), batch_size=batch_size)
    train_dl_trg = DataLoader(TensorDataset(target_words[:n_train]), batch_size=batch_size)

    val_dl_src = DataLoader(TensorDataset(sentences[n_train:]), batch_size=batch_size)
    val_dl_trg = DataLoader(TensorDataset(target_words[n_train:]), batch_size=batch_size)

    return (train_dl_src, train_dl_trg), (val_dl_src, val_dl_trg)


def get_stock_data(stock_name_1, stock_name_2):
    path_data_1 = './data/energy_pruned.npy'
    path_names_1 = './data/energy_pruned_names.npy'

    path_data_2 = './data/tech_pruned.npy'
    path_names_2 = './data/tech_pruned_names.npy'

    data_1 = np.load(path_data_1)
    names_1 = np.load(path_names_1)

    data_2 = np.load(path_data_2)
    names_2 = np.load(path_names_2)

    stock_idx_1 = np.where(names_1 == stock_name_1)[0][0]
    stock_data_1 = data_1[stock_idx_1]

    stock_data_2 = None
    if stock_name_2 is not None:
        stock_idx_2 = np.where(names_2 == stock_name_2)[0][0]
        stock_data_2 = data_2[stock_idx_2]

    return stock_data_1, stock_data_2


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    total_tokens = 0
    total_loss = 0
    tokens = 0

    start = time.time()

    for i, batch in enumerate(data_iter):
        # print('batch num:', i)
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
    #         if i % 50 == 0:
    #             elapsed = time.time() - start
    #             print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
    #                     (i, loss / batch.ntokens, tokens / elapsed))
    #             start = time.time()
    #             tokens = 0
    return total_loss / total_tokens


def train(model, train_dls, val_dls, vocab_size, num_epochs):
    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.1)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0.0001))  # lr=0.000001

    train_loss_avg_list = []
    val_loss_avg_list = []

    train_dl_src, train_dl_trg = train_dls
    val_dl_src, val_dl_trg = val_dls

    for epoch in range(num_epochs):
        model.train()  # set weights to training mode
        #         print('Training epoch...')
        train_loss_avg = run_epoch(yield_batch(train_dl_src, train_dl_trg), model,
                                   SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        #         print('Validation epoch...')
        val_loss_avg = run_epoch(yield_batch(val_dl_src, val_dl_trg), model,
                                 SimpleLossCompute(model.generator, criterion, None))

        val_loss_avg = float(val_loss_avg)
        print('Epoch: %i   Training Loss: %f   Validation Loss: %f' % (epoch, train_loss_avg, val_loss_avg))

        train_loss_avg_list.append(train_loss_avg)
        val_loss_avg_list.append(val_loss_avg)
    return train_loss_avg_list, val_loss_avg_list


def save_model(encoder_decoder, path):
    torch.save(encoder_decoder, path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=0)
    parser.add_argument("--l", type=int, default=0)
    parser.add_argument("--a", type=int, default=0)
    parser.add_argument("--c", type=int, default=0)

    parser.add_argument("--vocab-size", "-V", type=int, default=100)
    parser.add_argument("--num-encoder-layers", "-N", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", "-n", type=int, default=100)

    parser.add_argument("--save-dir", type=str, default='results')
    parser.add_argument("--data-source", type=str, default='energy')
    parser.add_argument("--stock-name-1", type=str, default='APA')
    parser.add_argument("--stock-name-2", type=str, default='BKR')
    parser.add_argument("--energy-to-tech", type=int, default=1)
    parser.add_argument("--gpu-idx", type=int, default=0)

    args = parser.parse_args()
    #     print(args)

    #     assert not (args.addition and args.concatenation)
    #     assert (args.addition or args.concatenation)
    #     assert not (args.sinusoidal and args.learnable)
    #     assert (args.sinusoidal or args.learnable)

    prefix = ''
    encoding_mode = ''
    combining_mode = ''

    if args.s and args.a:
        prefix = 'sa'
        encoding_mode = 'sinusoidal'
        combining_mode = 'add'
    elif args.l and args.a:
        prefix = 'la'
        encoding_mode = 'learnable'
        combining_mode = 'add'
    elif args.s and args.c:
        prefix = 'sc'
        encoding_mode = 'sinusoidal'
        combining_mode = 'concat'
    elif args.l and args.c:
        prefix = 'lc'
        encoding_mode = 'learnable'
        combining_mode = 'concat'

    save_dir = f'./results/{prefix}/'
    print(args.stock_name_1, args.stock_name_2, save_dir)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    print('Creating model...')
    model = make_model(args.vocab_size, args.vocab_size, N=args.num_encoder_layers, encoding_mode=encoding_mode,
                       combining_mode=combining_mode, max_wavelength=10000).to(DEVICE)

    print('Fetching data...')
    if args.energy_to_tech:
        stock_data_1, stock_data_2 = get_stock_data(args.stock_name_1, args.stock_name_2)
    else:
        stock_data_2, stock_data_1 = get_stock_data(args.stock_name_1, args.stock_name_2)

    if stock_data_2 is None:
        train_dls, val_dls = batch_data_price_prediction(stock_data_1, vocab_size=args.vocab_size,
                                                         batch_size=args.batch_size)
    else:
        print('TRANSLATION')
        train_dls, val_dls = batch_data_price_translation(stock_data_1, stock_data_2, vocab_size=args.vocab_size,
                                                          batch_size=args.batch_size)

    print('Starting Training...')
    train_loss, val_loss = train(model, train_dls, val_dls, vocab_size=args.vocab_size, num_epochs=args.num_epochs)
    print('Saving results...')
    np.save(save_dir + 'train_loss', np.array(train_loss))
    np.save(save_dir + 'val_loss', np.array(val_loss))
#     save_model(model, f'{save_dir}/model_trained.pt')