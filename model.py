import torch
from torch import nn

from torch.nn import functional as F
import math

class Embed(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embed, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    # handle subword encodings
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class TransformerSentiment(nn.Module):
    def __init__(self, vocab, d_embed, hidden_d, layers, nhead = 2, dropout=.2):
        super(TransformerSentiment, self).__init__()

        self.embed = Embed(d_embed, vocab)

        self.pos_embed = PositionalEncoding(d_embed, max_len=150, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_embed, nhead=nhead, dim_feedforward=hidden_d, dropout=dropout)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Linear(d_embed, 2)
        self.mask = None


        initrange = 0.1
        self.embed.lut.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, seq):

        if self.mask is None or self.mask.shape != seq.shape:
            self.mask = torch.zeros(seq.shape).to(seq.device)
    
        seq_mask = self.mask == seq

        emb = self.embed(seq)

        seq = self.pos_embed(emb).transpose(0, 1)

        seq = self.transformer(seq, src_key_padding_mask=seq_mask).transpose(0,1)

        B = seq.shape[0]
        S = seq.shape[1]
        out = self.fc(seq)
        return out

## Retrieved from pytorch website
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class LSTM(nn.Module):

    def __init__(self, embed_d, hidden_d, layers, bid=True):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(embed_d, hidden_d, layers,  bidirectional=True)

    def forward(self, seq):
        output, _ = self.lstm(seq.transpose(0,1))
        return output.transpose(0,1)

class FinalLinear(nn.Module):
    def __init__(self, tot_in_d=4, out_d=2):
        super(FinalLinear, self).__init__()

        self.fc = nn.Linear(tot_in_d, out_d)

    def forward(self, feats):
        return self.fc(feats)

class LSTMSentiment(nn.Module):
    def __init__(self, vocab, d_embed, hidden_d, layers, bid=True):
        super(LSTMSentiment, self).__init__()
        self.hidden_d = hidden_d
        self.embed = Embed(d_embed, vocab)
        self.lstm = LSTM(d_embed, int(hidden_d/2), layers, bid=True)
        if bid == True:
            self.fc = FinalLinear(hidden_d + 3, 2)
        else:
            self.fc = FinalLinear(hidden_d + 3, 2)
    
    def forward(self, seq, sent_1_hot):
        seq = self.lstm(self.embed(seq))

        seq = torch.cat((seq, sent_1_hot.float()), 2)
        out = []
        for i, s in enumerate(seq):
            out.append(self.fc(s.view(-1, self.hidden_d+3)).unsqueeze(0))
        return torch.cat(out)
        # return self.fc(torch.cat((seq, sent_1_hot.float()), 1))
