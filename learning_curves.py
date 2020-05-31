from data import Tweets
from model import Embed, LSTM, LSTMSentiment, TransformerSentiment

from utils import one_hot, jaccard

import torch
from torch import nn
import numpy as np

from curves import save_plots
import argparse

from train import fit


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_folds', default=5, type=int, help="Type of model to train [default: ./Data/ptb]")
    parser.add_argument('--model_type', default='blstm', help="Type of model to train [default: ./Data/ptb]")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size [default: 32]")
    parser.add_argument("--embed", default=64, type=int, help="Number of neurons of word embedding [default: 200]")
    parser.add_argument("--hidden", default=64, type=int, help="Number of neurons of each hidden layer [default: 200]")

    parser.add_argument("--layers", default=2, type=int, help="Number of encoder layers layers [default: 1]")

    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate [default: 20]")
    parser.add_argument("--epoch", default=100, type=int, help="Number of training epochs [default: 10]")
    parser.add_argument("--device", type=int, help="GPU card ID to use (if not given, use CPU)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed [default: 42]")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model_prefix = 'lc_{}_{}_{}_{}'.format(args.model_type, args.embed, args.hidden, args.layers)
    save = './{}_fold.pth'.format(model_prefix)

    device = 'cuda:{}'.format(args.device) if args.device is not None else 'cpu'

    dataset = Tweets(device)

    train_idx, val_idx = dataset.get_splits()

    percentages = [(1.0/7), (2.0/7), (3.0/7), (4.0/7), (5.0/7), (6.0/7), 1]
    
    train_losses = []
    test_losses = []


    train_accs = []
    test_accs = []
    metrics = []
    for p in percentages:
        tidx = train_idx[np.random.permutation(int(p * train_idx.shape[0]))]

        best = fit(args.model_type, dataset, tidx, val_idx, device, save, args, stopping=8)
        print(best)
        metrics.append(best)
    print(metrics)
    # n = len(train_losses)
    # xs = np.arange(n)

    # # plot losses
    # fig, ax = plt.subplots()
    # ax.plot(xs, train_losses, '--', linewidth=2, label='train')
    # ax.plot(xs, test_losses, '-', linewidth=2, label='validation')
    # ax.set_xlabel("Epoch")
    # ax.set_ylabel("Training Loss")
    # ax.legend(loc='upper right')
    # plt.savefig('{}loss.png'.format(model_prefix + '_'))

    # # plot train and test accuracies
    # plt.clf()
    # fig, ax = plt.subplots()
    # ax.plot(xs, train_accs, '--', linewidth=2, label='train')
    # ax.plot(xs, test_accs, '-', linewidth=2, label='validation')
    # ax.set_xlabel("Train Set Percentage")
    # ax.set_ylabel("Validation Accuracy")
    # ax.legend(loc='lower right')
    # plt.savefig('{}accuracy.png'.format(model_prefix + '_'))

    # plt.close()