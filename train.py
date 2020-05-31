from data import Tweets
from model import Embed, LSTM, LSTMSentiment, TransformerSentiment

from utils import one_hot, jaccard

import torch
from torch import nn
import numpy as np

from curves import save_plots
import argparse

def prepare_model(model_type, V, d_embed, d_lstm, layers, lr=.001, l2=0, device = 'cpu'):
    # model = 

    model = None
    if model_type == 'transformer':
        print(model_type)
        model = TransformerSentiment(V, d_embed, d_lstm, layers, nhead=4, dropout=.2).to(device)
    elif model_type == 'blstm':
        model = LSTMSentiment(V, d_embed, d_lstm, layers).to(device)
    
    criterion = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    return model, criterion, optimizer

def train(epoch, dataset, indices, batch_size, model, criterion, optimizer, device):
    r"""Train an epoch

    Args
    ----
    dataset : CoraDataset
        Cora dataset.
    indices : np.array
        An array of points ID to train on.
    model : torch.nn.Module
        Neural network model.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Training parameter optimizer.

    """
    model.train()

    indices = indices[np.random.permutation(indices.shape[0])]
    indices = indices[0:int(indices.shape[0]/batch_size)*batch_size]
    indices = indices.reshape(int(indices.shape[0]/batch_size), batch_size)

    count = 0
    loss_total = 0
    for batch_idx in indices:
        
        batch = dataset[list(batch_idx)]

        optimizer.zero_grad()
        inputs = []
        labels = []
        sentiments = []
        loss = None
        
        for seq_x, seq_y, _, _, sentiment in batch:

            inputs.append(seq_x.unsqueeze(0))
            labels.append(seq_y.unsqueeze(0))
            sentiments.append(sentiment.unsqueeze(0))

            # print('\r[Epoch {}][Batch {}/{}] Data to GPU'.format(epoch, count, len(indices)), end = '')
            # seq_x = [torch.LongTensor(word).to(device) for word in seq_x]
            # sentiment = one_hot([sentiment] * len(seq_x), 3).to(device)
            # print('\r[Epoch {}][Batch {}/{}] Forward'.format(epoch, count, len(indices)), end = '')
           
            
            # outputs.append(y_hat)
            # labels
        # print(outputs)

        # print(labels)
        # loss = criterion(torch.cat(outputs, 0), torch.cat(labels, 0))
        
        print('\r[Epoch {}][Batch {}/{}] Forward pass...'.format(epoch, count, len(indices)), end = '')
        y_hat = model(torch.cat(inputs), torch.cat(sentiments))#.reshape(len(batch), dataset.pad, -1)
        
        loss = 0

        for i, seq in enumerate(y_hat):
            l = criterion(seq.view(-1, 2), labels[i].view(-1))
            loss += l
            loss_total += l
        
        loss = loss / batch_size

        
        print('\r[Epoch {}][Batch {}/{}] Backward pass...'.format(epoch, count, len(indices)), end = '')
        loss.backward()
        optimizer.step()
        count += 1
        print('\r[Epoch {}][Batch {}/{}]'.format(epoch, count, len(indices)), end = '')

    return loss_total.data.item() / (len(indices) * batch_size)

def evaluate(epoch, dataset, indices, model, criterion, device):
    model.eval()
    with torch.no_grad():
        jaccard_sum, loss_sum, count = 0.0, 0.0, 0.0
        
    
        print('\r[Epoch {}][Evaluate {}/{}]'.format(epoch, count, len(indices)), end = '')
        for b in range(0, len(indices), 64):

            batch = indices[b:b+64]
            inputs = []
            labels = []
            sentiments = []
            for i in batch:
                seq_x, seq_y, _, _, sentiment = dataset[i]
                inputs.append(seq_x.unsqueeze(0))
                labels.append(seq_y.unsqueeze(0))
                sentiments.append(sentiment.unsqueeze(0))

            y_hat_full = model(torch.cat(inputs), torch.cat(sentiments))#.reshape(len(batch), dataset.pad, -1)

            for i, seq in enumerate(y_hat_full):
                loss_sum += criterion(seq.view(-1, 2), labels[i].view(-1))
            
            
            y_hat = torch.argmax(y_hat_full, dim=2)
            final = dataset.tokenizer.decode_batch((y_hat * torch.cat(inputs)).tolist())
            for row, i in enumerate(batch):
                seq_x, seq_y, tweet, selection, sentiment = dataset[i]

                selection_output = final[row]

                jaccard_sum += jaccard(' '.join(selection), ' '.join(selection_output))
                count += 1
                print('\r[Epoch {}][Evaluate {}/{}]'.format(epoch, count, len(indices)), end = '')
        return loss_sum.data.item() / count, jaccard_sum / count

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

    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate [default: 20]")
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

    train_idx, val_idx = dataset.get_splits()

    V = len(dataset.vocab.keys())

    model, criterion, optimizer = prepare_model(args.model_type, V, args.embed, args.hidden, args.layers, lr=args.lr, device = device)
    

    train_losses = []
    train_jaccs = []
    val_losses = []
    val_jaccs = []

    best_jacc = 0
    
    for i in range(args.epoch):

        train_loss = train(i, dataset, train_idx, args.batch_size, model, criterion, optimizer, device)
        
        # train_loss = 0
        strain_loss, train_jacc = evaluate(i, dataset, train_idx[np.random.permutation(8000)], model, criterion, device)
        val_loss, val_jacc = evaluate(i, dataset, val_idx, model, criterion, device)

        train_losses.append(train_loss)
        train_jaccs.append(train_jacc)
        val_losses.append(val_loss)
        val_jaccs.append(val_jacc)

        if val_jacc > best_jacc:
            best_jacc = val_jacc
            torch.save(model.state_dict(), save)
        print("\r[Epoch {}] Train Loss: {}, Train Jaccard: {} | Val Loss: {}, Val Jaccard: {}".format(i, train_loss, train_jacc, val_loss, val_jacc))

        save_plots(train_losses, train_jaccs, val_losses, val_jaccs, file_prefix=model_prefix)
