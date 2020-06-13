from data import Tweets
from model import Embed, LSTM, LSTMSentiment, TransformerSentiment

from utils import one_hot, jaccard

import torch
from torch import nn
import numpy as np

from curves import save_plots
import argparse

from train import fit


# def fit(model_type, dataset, train_idx, val_idx, device, args):

#     V = len(dataset.vocab.keys())
#     model, criterion, optimizer = prepare_model(args.model_type, V, args.embed, args.hidden, args.layers, device = device)


#     train_losses = []
#     train_jaccs = []
#     val_losses = []
#     val_jaccs = []

#     best_jacc = 0
#     best_index

#     for i in range(args.epoch):

#         train_loss = train(i, dataset, train_idx, args.batch_size, model, criterion, optimizer, device)


#         strain_loss, train_jacc = evaluate(i, dataset, train_idx[np.random.permutation(8000)], model, criterion, device)
#         val_loss, val_jacc = evaluate(i, dataset, val_idx, model, criterion, device)

#         train_losses.append(train_loss)
#         train_jaccs.append(train_jacc)
#         val_losses.append(val_loss)
#         val_jaccs.append(val_jacc)

#         if val_jacc > best_jacc:
#             torch.save(model.state_dict(), save)
#         print("\r[Epoch {}] Train Loss: {}, Train Jaccard: {} | Val Loss: {}, Val Jaccard: {}".format(i, train_loss, train_jacc, val_loss, val_jacc))


#     model.load_state_dict(torch.load(save))

#     train_loss, train_jacc = evaluate(i, dataset, train_idx, model, criterion, device)
#     val_loss, val_jacc = evaluate(i, dataset, val_idx, model, criterion, device)
#     return train_loss, train_jacc, val_loss, val_jacc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_folds', default=5, type=int,
                        help="Type of model to train [default: ./Data/ptb]")
    parser.add_argument('--model_type', default='blstm',
                        help="Type of model to train [default: ./Data/ptb]")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size [default: 32]")
    parser.add_argument("--embed", default=64, type=int,
                        help="Number of neurons of word embedding [default: 200]")
    parser.add_argument("--hidden", default=64, type=int,
                        help="Number of neurons of each hidden layer [default: 200]")

    parser.add_argument("--layers", default=2, type=int,
                        help="Number of encoder layers layers [default: 1]")

    parser.add_argument("--lr", default=1e-3, type=float,
                        help="Learning rate [default: 20]")
    parser.add_argument("--epoch", default=100, type=int,
                        help="Number of training epochs [default: 10]")
    parser.add_argument("--device", type=int,
                        help="GPU card ID to use (if not given, use CPU)")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed [default: 42]")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_prefix = '{}_{}_{}_{}'.format(
        args.model_type, args.embed, args.hidden, args.layers)
    save = './{}_fold.pth'.format(model_prefix)

    device = 'cuda:{}'.format(
        args.device) if args.device is not None else 'cpu'

    dataset = Tweets(device)

    folds = dataset.k_folds(args.k_folds)

    metrics = []
    for i, fold in enumerate(folds):

        print("-------Fitting Fold K={}-------".format(i))
        val_idx = fold
        train_idx = []
        for j, fold in enumerate(folds):
            if i != j:
                train_idx.append(fold)

        train_idx = np.concatenate(train_idx)

        best = fit(args.model_type, dataset, train_idx,
                   val_idx, device, save, args, stopping=15)
        print(best)
        metrics.append(best)
    print()
    print("-------Final Metrics--------")
    print(metrics)

    mean = np.zeros(4)
    for m in metrics:
        mean += np.array(list(m))
    mean = mean / args.k_folds

    print("---------Average---------")
    print(mean)
    std = np.zeros(4)
    for m in metrics:
        std += (np.array(list(m)) - mean)**2
    std = np.sqrt(std/args.k_folds)
    print("---------STDEV---------")
    print(std)
