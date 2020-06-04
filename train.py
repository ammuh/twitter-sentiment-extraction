from data import Tweets
from model import Embed, LSTM, LSTMSentiment, TransformerSentiment

from utils import one_hot, jaccard

import torch
from torch import nn
import numpy as np

from curves import save_plots
import argparse
from tqdm import tqdm
import math

from torch.utils.data import Dataset, DataLoader


def prepare_model(model_type, V, d_embed, d_lstm, layers, nhead, dropout=.2, lr=.001, l2=0, device = 'cpu'):
    # model = 

    model = None
    if model_type == 'transformer':
        print(model_type)
        model = TransformerSentiment(V, d_embed, d_lstm, layers, nhead=nhead, dropout=dropout).to(device)
    elif model_type == 'blstm':
        model = LSTMSentiment(V, d_embed, d_lstm, layers).to(device)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-1)#size_average=True, ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    return model, criterion, optimizer

def train(epoch, dataset, dataloader, model, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    count = 0
    pbar = tqdm(desc='Train Batch', total=len(dataloader), leave=False)

    for i, batch in enumerate(dataloader):

        tweet = batch['tweet'].to(device)
        selection = batch['selection'].long().to(device)
        non_pad_elements = selection.shape[1] - (selection == -1).sum(dim=1)

        optimizer.zero_grad()

        y_hat = model(tweet)
        
        loss = criterion(y_hat.permute(0,2,1), selection)
        loss = (loss / non_pad_elements.unsqueeze(-1)).sum() / tweet.shape[0]
        
        loss_sum += loss.data.item()
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        count += 1
        pbar.update()
    pbar.clear()
    pbar.close()
    return loss_sum / count

def evaluate(epoch, dataset, dataloader, model, criterion, device, prefix=None):
    model.eval()
    with torch.no_grad():
        jaccard_sum, loss_sum, count = 0.0, 0.0, 0.0

        pbar = tqdm(desc='{}Eval Batch'.format('' if prefix is None else prefix + ' '), total=len(dataloader), leave=False)

        for i, batch in enumerate(dataloader):
            
            tweet = batch['tweet'].to(device)
            selection = batch['selection'].long().to(device)
            raw_selection = batch['raw_selection']

            non_pad_elements = selection.shape[1] - (selection == -1).sum(dim=1)
            y_hat = model(tweet)

            loss = criterion(y_hat.permute(0,2,1), selection)
            loss = (loss / non_pad_elements.unsqueeze(-1)).sum() / tweet.shape[0]
            loss_sum += loss.data.item()

            y_hat = torch.argmax(y_hat, dim=2)
            final = dataset.tokenizer.decode_batch((y_hat * tweet).tolist())
            # print(final)
            for j, raw in enumerate(raw_selection):
                selection_output = final[j]
                jaccard_sum += jaccard(raw, selection_output) / tweet.shape[0]
            
            count += 1
            pbar.update()
        pbar.clear()
        pbar.close()
        return loss_sum / count, jaccard_sum / count

def fit(model_type, dataset, train_idx, val_idx, device, save, args, stopping=4):

    V = len(dataset.vocab.keys())
    model, criterion, optimizer = prepare_model(args.model_type, V, args.embed, args.hidden, args.layers, lr=args.lr, device = device)
    

    train_losses = []
    train_jaccs = []
    val_losses = []
    val_jaccs = []

    best_jacc = 0
    
    break_loop = False
    prev_loss = float('inf')

    for i in range(args.epoch):

        train_loss = train(i, dataset, train_idx, args.batch_size, model, criterion, optimizer, device)
        
        
        strain_loss, train_jacc = evaluate(i, dataset, train_idx[np.random.permutation(min(8000, train_idx.shape[0]))], model, criterion, device)
        val_loss, val_jacc = evaluate(i, dataset, val_idx, model, criterion, device)

        train_losses.append(train_loss)
        train_jaccs.append(train_jacc)
        val_losses.append(val_loss)
        val_jaccs.append(val_jacc)

        badge = '[ ]'
        if i >= stopping and val_loss > prev_loss and val_jacc < best_jacc:
            break_loop = True
        
        if val_jacc > best_jacc:
            best_jacc = val_jacc
            torch.save(model.state_dict(), save)
            badge = '[*]'
        
        print("\r{}[Epoch {}] Train Loss: {}, Train Jaccard: {} | Val Loss: {}, Val Jaccard: {}".format(badge, i, train_loss, train_jacc, val_loss, val_jacc))
       
        prev_loss = val_loss

        if break_loop is True:
            break

    
    model.load_state_dict(torch.load(save))

    train_loss, train_jacc = evaluate('final', dataset, train_idx, model, criterion, device)
    val_loss, val_jacc = evaluate('final', dataset, val_idx, model, criterion, device)
    return train_loss, train_jacc, val_loss, val_jacc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='transformer', help="Type of model to train [default: ./Data/ptb]")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size [default: 32]")
    parser.add_argument("--embed", default=64, type=int, help="Number of neurons of word embedding [default: 200]")
    parser.add_argument("--hidden", default=64, type=int, help="Number of neurons of each hidden layer [default: 200]")
    parser.add_argument("--layers", default=2, type=int, help="Number of encoder layers layers [default: 1]")
    parser.add_argument("--nhead", default=2, type=int, help="Number of attention heads [default: 2]")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate [default: 20]")
    parser.add_argument("--dropout", default=.2, type=float, help="Dropout [default: 20]")
    parser.add_argument("--epoch", default=100, type=int, help="Number of training epochs [default: 10]")
    parser.add_argument("--device", type=int, help="GPU card ID to use (if not given, use CPU)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed [default: 42]")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model_prefix = '{}_{}_{}_{}'.format(args.model_type, args.embed, args.hidden, args.layers)
    save = './{}.pth'.format(model_prefix)

    device = 'cuda:{}'.format(args.device) if args.device is not None else 'cpu'

    dataset = Tweets(device)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [math.floor(len(dataset)*.7), math.floor(len(dataset)*.3)])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=12, pin_memory=True)

    V = len(dataset.vocab.keys())

    model, criterion, optimizer = prepare_model(args.model_type, V, args.embed, args.hidden, args.layers, args.nhead, dropout=args.dropout, lr=args.lr, device = device)
    
    pbar = tqdm(desc='Epoch', total=args.epoch)

    for i in range(args.epoch):
        pbar.write('--------------Epoch {}--------------'.format(i))
        train(i, dataset, train_dataloader,  model, criterion, optimizer, device)
        train_loss, train_jaccard = evaluate(i, dataset, train_dataloader, model, criterion, device)
        pbar.write('Train Loss: {}, Train Jaccard: {}'.format(train_loss, train_jaccard))
        val_loss, val_jaccard = evaluate(i, dataset, val_dataloader, model, criterion, device)
        pbar.write('Val Loss: {}, Val Jaccard: {}'.format(val_loss, val_jaccard))
        pbar.update()
    pbar.close()

    # train_losses = []
    # train_jaccs = []
    # val_losses = []
    # val_jaccs = []

    # best_jacc = 0
    
    # for i in range(args.epoch):

    #     train_loss = train(i, train_dataset, train_dataloader, model, criterion, optimizer, device)
        
    #     strain_loss, train_jacc = evaluate(i, dataset, train_idx, model, criterion, device)
    #     val_loss, val_jacc = evaluate(i, dataset, val_idx, model, criterion, device)

    #     train_losses.append(train_loss)
    #     train_jaccs.append(train_jacc)
    #     val_losses.append(val_loss)
    #     val_jaccs.append(val_jacc)

    #     if val_jacc > best_jacc:
    #         best_jacc = val_jacc
    #         torch.save(model.state_dict(), save)
    #     print("\r[Epoch {}] Train Loss: {}, Train Jaccard: {} | Val Loss: {}, Val Jaccard: {}".format(i, train_loss, train_jacc, val_loss, val_jacc))

    #     save_plots(train_losses, train_jaccs, val_losses, val_jaccs, file_prefix=model_prefix)
