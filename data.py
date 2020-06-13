import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
import tokenizers
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import torch
from utils import one_hot, jaccard
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.data import load

TRAIN_PATH = './data/train.csv'
TEST_PATH = './data/test.csv'


class Tweets(Dataset):
    def __init__(self, device='cpu', pad=150, test=False, N=4):
        self.samples = []
        self.pad = pad

        self.tokenizer = BertWordPieceTokenizer(
            "./data/bert-base-uncased-vocab.txt", lowercase=True, clean_text=True)

        self.tokenizer.enable_padding(
            max_length=pad-1)  # -1 for sentiment token

        self.tokenizer.add_special_tokens(['[POS]'])
        self.tokenizer.add_special_tokens(['[NEG]'])
        self.tokenizer.add_special_tokens(['[NEU]'])
        self.vocab = self.tokenizer.get_vocab()

        self.sent_t = {
            'positive': self.tokenizer.token_to_id('[POS]'),
            'negative': self.tokenizer.token_to_id('[NEG]'),
            'neutral': self.tokenizer.token_to_id('[NEU]')
        }

        self.pos_set = {
            'UNK': 0
        }
        all_pos = load('help/tagsets/upenn_tagset.pickle').keys()

        for i, p in enumerate(all_pos):
            self.pos_set[p] = i+1
                    

        print(self.pos_set)
        self.tweet_tokenizer = TweetTokenizer()

        data = None
        if test is True:
            data = pd.read_csv(TEST_PATH).values
            for row in data:
                tid, tweet, sentiment = tuple(row)

                pos_membership = [0] * len(tweet)

                pos_tokens = self.tweet_tokenizer.tokenize(tweet)
                pos = nltk.pos_tag(pos_tokens)
                offset = 0

                for i, token in enumerate(pos_tokens):
                    start = tweet.find(token, offset)
                    end = start + len(token)
                    if pos[i][1] in self.pos_set:
                        pos_membership[start:end] = [self.pos_set[pos[i][1]]] * len(token)
                    offset += len(token)

                tokens = self.tokenizer.encode(tweet)
                word_to_index = tokens.ids
                offsets = tokens.offsets
                
                token_pos = [0] * len(word_to_index)
                # get pos info
                for i, (s, e) in enumerate(offsets):
                    if word_to_index[i] == 0 or word_to_index[i] == 101 or word_to_index[i] == 102:
                        pass
                    elif s != e:
                        sub = pos_membership[s:e]
                        token_pos[i] = max(set(sub), key=sub.count)

                token_pos =  [0] + token_pos       
                word_to_index = [self.sent_t[sentiment]] + word_to_index
                offsets = [(0, 0)] + offsets
                offsets = np.array([[off[0], off[1]] for off in offsets])
                word_to_index = np.array(word_to_index)
                token_pos = np.array(token_pos)

                self.samples.append({
                    'tid': tid,
                    'sentiment': sentiment,
                    'tweet': word_to_index,
                    'offsets': offsets,
                    'raw_tweet': tweet,
                    'pos': token_pos
                })

        else:

            data = pd.read_csv(TRAIN_PATH).values
            data = augment_n(data, N=N)

            for row in data:
                tid, tweet, selection, sentiment = tuple(row)

                char_membership = [0] * len(tweet)
                pos_membership = [0] * len(tweet)
                si = tweet.find(selection)
                if si < 0:
                    char_membership[0:] = [1] * len(char_membership)
                else:
                    char_membership[si:si +
                                    len(selection)] = [1] * len(selection)

                pos_tokens = self.tweet_tokenizer.tokenize(tweet)
                pos = nltk.pos_tag(pos_tokens)
                offset = 0

                for i, token in enumerate(pos_tokens):
                    start = tweet.find(token, offset)
                    end = start + len(token)
                    if pos[i][1] in self.pos_set:
                        pos_membership[start:end] = [self.pos_set[pos[i][1]]] * len(token)
                    offset += len(token)
                
                tokens = self.tokenizer.encode(tweet)
                word_to_index = tokens.ids
                offsets = tokens.offsets

                token_membership = [0] * len(word_to_index)
                token_pos = [0] * len(word_to_index)

                # Inclusive indices
                start = None
                end = None
                for i, (s, e) in enumerate(offsets):
                    if word_to_index[i] == 0 or word_to_index[i] == 101 or word_to_index[i] == 102:
                        token_membership[i] = -1
                    elif sum(char_membership[s:e]) > 0:
                        token_membership[i] = 1
                        if start is None:
                            start = i+1
                        end = i+1

                # get pos info
                for i, (s, e) in enumerate(offsets):
                    if word_to_index[i] == 0 or word_to_index[i] == 101 or word_to_index[i] == 102:
                        pass
                    elif s != e:
                        sub = pos_membership[s:e]
                        token_pos[i] = max(set(sub), key=sub.count)

                if start is None:
                    print(tweet)
                    print(selection)
                    continue
                # token_membership = torch.LongTensor(token_membership).to(device)
                word_to_index = [self.sent_t[sentiment]] + word_to_index
                token_membership = [-1] + token_membership
                offsets = [(0, 0)] + offsets
                token_pos =  [0] + token_pos                
                
                offsets = np.array([[off[0], off[1]] for off in offsets])
                word_to_index = np.array(word_to_index)
                token_membership = np.array(token_membership).astype('float')
                token_pos = np.array(token_pos)

                if tid is None:
                    raise Exception('None field detected')
                if sentiment is None:
                    raise Exception('None field detected')
                if word_to_index is None:
                    raise Exception('None field detected')
                if token_membership is None:
                    raise Exception('None field detected')
                if selection is None:
                    raise Exception('None field detected')
                if tweet is None:
                    raise Exception('None field detected')
                if start is None:
                    raise Exception('None field detected')
                if end is None:
                    raise Exception('None field detected')
                if offsets is None:
                    raise Exception('None field detected')

                self.samples.append({
                    'tid': tid,
                    'sentiment': sentiment,
                    'tweet': word_to_index,
                    'selection': token_membership,
                    'raw_selection': selection,
                    'raw_tweet': tweet,
                    'start': start,
                    'end': end,
                    'offsets': offsets,
                    'pos': token_pos
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


def stop_words():
    words = []
    single_chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
    for c in single_chars:
        words.append(str(c))
    nltk_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                  'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
    words = words + nltk_words
    return words


def augment_n(data, N=1):
    pbar = tqdm(desc='Augmenting Data N={}'.format(N),
                total=data.shape[0], leave=False)

    # random synonym replacement
    aug = naw.SynonymAug(aug_max=4, stopwords=stop_words())
    results = []
    for row in data:
        t, s = augment(row[1], row[2], aug, N)
        augs = []

        for j, t in enumerate(t):
            augs.append([row[0] + str(j), t, s[j], row[3]])
        results.append(np.array(augs))
        pbar.update()

    results.append(data)
    pbar.clear()
    pbar.close()
    return np.concatenate(results)


def augment(tweet, selection, aug, n):
    # TODO: random character insertion
    # TODO: random character repitition
    # TODO: random character replacement
    tweet = tweet.lower()
    selection = selection.lower()

    start = tweet.find(selection)
    end = start + len(selection)

    tweet = tweet[0:start] + '#>' + tweet[start:end] + '#>' + tweet[end:]

    results = aug.augment(tweet, n=n)

    tweets = []
    selections = []

    for aug_tweet in results:
        p_start = aug_tweet.find('# >')

        p_end = aug_tweet.find('# >', p_start+3)

        if p_start == -1:
            p_start = aug_tweet.find('#>')
             
            p_end = aug_tweet.find('#>', p_start+2)

        aug_tweet = aug_tweet[0:p_start] + \
            aug_tweet[p_start+4:p_end] + aug_tweet[p_end+4:]
        tweets.append(aug_tweet)
        aug_selection = aug_tweet[p_start:p_end - 4]
        selections.append(aug_selection)

    return tweets, selections


if __name__ == "__main__":

    # aug = naw.SynonymAug(include_detail=True)

    # print(len(aug.augments(["Yo man what's up how is everything", "This method will augment my text"], n=4)))
    # print(augment("hello this is a cool fucking tweet maaaaan!", "cool fucking tweet", aug, 5))
    dataset = Tweets()
    print(dataset[0])
    # print(len(dataset))
    # print(dataset[2])
    # dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

    # for data in dataloader:

    #     print(data['tweet'])
    #     print(data['start'])
    #     print(data['end'])
    #     break
