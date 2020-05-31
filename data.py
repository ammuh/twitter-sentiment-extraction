import pandas as pd 
import numpy as np
import re
from torch.utils.data import Dataset
from tokenizers import BertWordPieceTokenizer
import tokenizers

import torch
from utils import one_hot, jaccard


def subarray(arr, sub):
    if arr is None or sub is None:
        return None

    if len(sub) == 0 or len(sub) > len(arr):
        return False
    
    if len(sub) == len(arr):
        return arr == sub
    
    for i in range(0, len(arr) - len(sub), 1):
        if arr[i:i+len(sub)] == sub:
            return True
    
    return False

def subarray_index(arr, sub):
    highlight = [0.0 for i in range(len(arr))]

    first = sub[0]
    if first not in arr:
        sub.pop(0)

    if len(sub) == 0:
        return highlight
    
    last = sub[-1]
    if last not in arr:
        sub.pop()
    
    if len(sub) == 0:
        return highlight
    

    if len(sub) == len(arr):
        if arr == sub:
            highlight[:] = [1.0] * len(highlight)
        return highlight

    for i in range(0, len(arr) - len(sub), 1):
        if arr[i:i+len(sub)] == sub:
            highlight[i:i+len(sub)] = [1.0] * len(sub)
            break
    
    return highlight


def isLink(text):
    return len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)) != 0



class Tweets(Dataset):
    def __init__(self, device='cpu', pad=150, test=False):
        self.samples = []
        self.pad = pad
        
        self.tokenizer = BertWordPieceTokenizer("./data/bert-large-uncased-vocab.txt", lowercase=True, clean_text=True)

        # self.tokenizer.add_tokens(['[LINK]'])
        # self.tokenizer.add_tokens(['[MENTION]'])
        # self.tokenizer.add_tokens(['[HASH]'])
        # self.tokenizer.add_tokens(['[NUM]'])
        self.tokenizer.enable_padding(max_length = pad)
        self.vocab = self.tokenizer.get_vocab()

        self.sent_i = {
            'positive' : 0,
            'negative' : 1,
            'neutral' : 2
        }

        data = None
        if test is True:
            data = pd.read_csv('./data/test.csv').values
            for row in data:
                tid, tweet, sentiment = tuple(row)
                

                tokens = self.tokenizer.encode(tweet)
                word_to_index = tokens.ids

            
                self.samples.append((
                    tid,
                    torch.LongTensor(word_to_index).to(device), #[torch.LongTensor(word).to(device) for word in word_to_index], 
                    one_hot([self.sent_i[sentiment]] * pad, 3).to(device)
                ))
        else:
            data = pd.read_csv('./data/train.csv').values

            

            # word_counter = 0
            # for row in data:
            #     tid, tweet, selection, sentiment = tuple(row)
            #     tweet = tweet.lower().split()
            #     selection = selection.lower().split()
                
            #     for word in tweet:
            #         if self.transform(word) not in self.vocab:
            #             self.vocab[self.transform(word)] = word_counter
            #             word_counter += 1
                # print(tweet)
                # print(selection)
                
                # if not (subarray(tweet, selection) or subarray(tweet, selection[1:])):
                #     print(tweet)
                #     print(selection)
                # print(tid, tweet, selection, sentiment)
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
                    if word_to_index[i] == 0:
                        token_membership[i] = -1
                    elif sum(char_membership[start:end]) > 0:
                        token_membership[i] = 1

                token_membership = torch.LongTensor(token_membership).to(device)

                self.samples.append((
                    torch.LongTensor(word_to_index).to(device), #[torch.LongTensor(word).to(device) for word in word_to_index], 
                    token_membership, 
                    tweet, 
                    selection, 
                    one_hot([self.sent_i[sentiment]] * pad, 3).to(device)
                ))
            # selection = selection.lower().split()
            # tweet = tweet.lower().split()
            # # word_to_index = [[101]]

            # # for word in tweet:
            # #     clean = self.transform(word)
            # #     cleaned.append(clean)
            # #     ids = self.tokenizer.encode(clean).ids
            # #     ids = ids[1:-1]
            # #     word_to_index.append(ids)

            # # word_to_index.append([102])

            # self.samples.append((
            #     torch.LongTensor(word)#[torch.LongTensor(word).to(device) for word in word_to_index], 
            #     torch.LongTensor([0] + subarray_index(tweet, selection) + [0]).to(device), 
            #     tweet, 
            #     selection, 
            #     one_hot([self.sent_i[sentiment]] * len(word_to_index), 3).to(device)
            # ))
            
    def transform(self, w):
        # check if link
        orig = w

        # # Replace links
        # w = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', lambda x: '[LINK]', w)

        # # handle mentions
        # w = re.sub(r'[@](\w+)', '[MENTION]\1', w)
        
        # # handle hashtags
        # w = re.sub(r'[#](\w+)', '[HASH]\1', w)

        # # replace numbers
        # w = re.sub('[0-9]+', lambda x: '[NUM]', w)

        # # reduce repeated characters over 3
        # w = re.sub('(([a-z])\2\2)\2+', lambda x: x[0] + x[1] + x[2], w)

        # Clean up everything else
        # w = re.sub('[^A-Za-z0-9\[\]]', lambda x: '', w)
        return w
    
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
    print(dataset.k_folds(5))
    print(dataset[0])
    # mlen = 0
    # for data in dataset:
    #     mlen = max(mlen, len(data[2]))
    # print(mlen)

    # data = pd.read_csv('./data/train.csv').values

    # text = ""
    # for row in data:
    #     tid, tweet, selection, sentiment = tuple(row)
    #     text += tweet + '\n'

    # file1 = open("./data/train_corpus.txt","w")
    # file1.write(text) 
    # file1.close()
