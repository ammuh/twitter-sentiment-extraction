import pandas as pd 
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
import tokenizers

import torch
from utils import one_hot, jaccard

TRAIN_PATH = './data/train.csv'
TEST_PATH = './data/test.csv'

class Tweets(Dataset):
    def __init__(self, device='cpu', pad=150, test=False):
        self.samples = []
        self.pad = pad
        
        self.tokenizer = BertWordPieceTokenizer("./data/bert-base-uncased-vocab.txt", lowercase=True, clean_text=True)

        self.tokenizer.enable_padding(max_length = pad-1) # -1 for sentiment token
        self.vocab = self.tokenizer.get_vocab()
        self.tokenizer.add_special_tokens(['[POS]'])
        self.tokenizer.add_special_tokens(['[NEG]'])
        self.tokenizer.add_special_tokens(['[NEU]'])

        self.sent_t = {
            'positive' : self.tokenizer.token_to_id('[POS]'),
            'negative' : self.tokenizer.token_to_id('[NEG]'),
            'neutral' : self.tokenizer.token_to_id('[NEU]')
        }

        data = None
        if test is True:
            data = pd.read_csv(TEST_PATH).values
            for row in data:
                tid, tweet, sentiment = tuple(row)
                

                tokens = self.tokenizer.encode(tweet)
                word_to_index = [self.sent_t[sentiment]] + tokens.ids

                
                self.samples.append({
                    'tid' : tid, 
                    'sentiment' : sentiment,
                    'tweet' : word_to_index
                })

        else:
            
            data = pd.read_csv(TRAIN_PATH).values

            for row in data:
                tid, tweet, selection, sentiment = tuple(row)
                
                char_membership = [0] * len(tweet)
                si = tweet.find(selection)
                
                char_membership[si:si+len(selection)] = [1] * len(selection)

                tokens = self.tokenizer.encode(tweet)
                word_to_index = tokens.ids
                offsets = tokens.offsets

                token_membership = [0] * len(word_to_index)

                for i, (start, end) in enumerate(offsets):
                    if word_to_index[i] == 0 or word_to_index[i] == 101 or word_to_index[i] == 102:
                        token_membership[i] = -1
                    elif sum(char_membership[start:end]) > 0:
                        token_membership[i] = 1

                # token_membership = torch.LongTensor(token_membership).to(device)
                word_to_index = [self.sent_t[sentiment]] + word_to_index
                token_membership = [-1] + token_membership

                word_to_index = np.array(word_to_index)
                token_membership = np.array(token_membership).astype('float')

                self.samples.append({
                    'tid' : tid, 
                    'sentiment' : sentiment,
                    'tweet' : word_to_index,
                    'selection' : token_membership
                })
              
    def get_splits(self, val_size=.3):
        N = len(self.samples)
        indices = np.random.permutation(N)
        split = int(N * (1-val_size))
        train_indices = indices[0:split]
        valid_indices = indices[split:]
        return train_indices, valid_indices

    def k_folds(self, k=5):
        N = len(self.samples)
        indices = np.random.permutation(N)
        return np.array_split(indices, k)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            return self.samples[idx]
        except TypeError:
            pass
        return [self.samples[i] for i in idx]
        

if __name__ == "__main__":

    dataset = Tweets()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    for data in dataloader:
        print(data['tweet'])
        break