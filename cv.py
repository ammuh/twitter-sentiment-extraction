from data import Tweets
from model import Embed, TransformerSentiment
import time
from utils import one_hot, jaccard
from torch.utils.data import Subset, DataLoader
import torch
from torch import nn
import json
import numpy as np

from curves import save_plots
import argparse

from train import fit, prepare_model


if __name__ == "__main__":
    configs = [
        # {
        #     'augment_n': 4,
        #     'batch_size' : 100,
        #     'embed' : 128,
        #     'layers' : 4,
        #     'hidden' : 128,
        #     'nhead' : 8,
        #     'dropout' : 0.2,
        #     'label_smoothing': .1,
        #     'lr' : 5e-4,
        #     'epoch' : 50,
        #     'device' : 0
        # },
        # {
        #     'augment_n': 4,
        #     'batch_size' : 100,
        #     'embed' : 128,
        #     'layers' : 8,
        #     'hidden' : 128,
        #     'nhead' : 8,
        #     'dropout' : 0.2,
        #     'label_smoothing': .1,
        #     'lr' : 5e-4,
        #     'epoch' : 65,
        #     'device' : 0
        # },
        {
            'augment_n': 4,
            'batch_size' : 100,
            'embed' : 128,
            'layers' : 12,
            'hidden' : 128,
            'nhead' : 8,
            'dropout' : 0.2,
            'label_smoothing': .1,
            'lr' : 5e-4,
            'epoch' : 80,
            'device' : 0
        }
    ]

    torch.manual_seed(42)
    np.random.seed(42)

    for args in configs:
        
        uid = int(time.time())
        print('----------Config {}----------'.format(uid))
        print(args)
        model_prefix = '{}'.format(uid)

        with open('./config/{}.json'.format(model_prefix), 'w') as fp:
            json.dump(args, fp)

        device = 'cuda:{}'.format(
            args['device']) if args['device'] is not None else 'cpu'

        dataset = Tweets(device, N=args['augment_n'])
        folds = dataset.k_folds(5)

        fold_stats = []

        for i, fold in enumerate(folds):
            tr = []
            for j, f in enumerate(folds):
                if i != j:
                    tr.append(f)
            
            train_dataset = Subset(dataset, np.concatenate(tr))
            val_dataset = Subset(dataset, fold)
            train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
            val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], num_workers=12, pin_memory=True)
            V = len(dataset.vocab.keys())
            P = len(dataset.pos_set.keys())

            model, criterion, optimizer = prepare_model(
                V, 
                P, 
                args['embed'],
                args['hidden'], 
                args['layers'], 
                args['nhead'], 
                dropout=args['dropout'],
                smoothing=args['label_smoothing'], 
                lr=args['lr'], 
                device=device    
            )

            best_loss, best_jacc = fit(model, train_dataloader, val_dataloader, criterion, optimizer, device, args['epoch'], model_prefix + '_' + str(i))
            fold_stats.append([best_loss, best_jacc])

            print('Fold {} - Best Loss: {}, Best Jacc: {}'.format(i, best_loss, best_jacc))
        fold_stats = np.array(fold_stats)
        mean = np.mean(fold_stats, axis=0)
        
        std = np.std(fold_stats, axis=0)

        print(mean)
        print(std)

