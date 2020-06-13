import torch
from torch import nn

from torch.nn import functional as F
import math
import torch


# pre_trained = torch.hub.load(
#     'huggingface/pytorch-transformers', 'model', 'bert-base-uncased')


class Embed(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embed, self).__init__()
        # self.lut = pre_trained.embeddings.word_embeddings
        self.lut = nn.Embedding(vocab, d_model)
        self.lut.weight.data.uniform_(-0.1, .1)
        # v_trained = pre_trained.embeddings.word_embeddings.weight.shape[0]
        # self.lut.weight.data[0:v_trained] = pre_trained.embeddings.word_embeddings.weight
        self.d_model = d_model

        # for param in self.lut.parameters():
        #     param.requires_grad = False

    # handle subword encodings
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class TransformerSentiment(nn.Module):
    def __init__(self, vocab, pos, d_embed, hidden_d, layers, nhead=2, dropout=.2):
        super(TransformerSentiment, self).__init__()

        self.embed = Embed(d_embed, vocab)
        self.pos = Embed(d_embed, pos)

        self.pos_embed = PositionalEncoding(
            d_embed, max_len=150, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_embed, nhead=nhead, dim_feedforward=hidden_d, dropout=dropout)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.fc_start = nn.Linear(150*d_embed, 150)
        self.fc_end = nn.Linear(150*d_embed, 150)

        self.mask = None

        initrange = 0.1

        self.fc_start.bias.data.zero_()
        self.fc_start.weight.data.uniform_(-initrange, initrange)
        self.fc_end.bias.data.zero_()
        self.fc_end.weight.data.uniform_(-initrange, initrange)

    def forward(self, seq, pos):

        if self.mask is None or self.mask.shape != seq.shape:
            self.mask = torch.zeros(seq.shape).to(seq.device)

        seq_mask = self.mask == seq
        emb = self.embed(seq)
        pos = self.pos(pos)

        seq = self.pos_embed(emb + pos).transpose(0, 1)
        seq = self.transformer(
            seq, src_key_padding_mask=seq_mask).transpose(0, 1)
        B = seq.shape[0]
        S = seq.shape[1]

        # start = self.fc_start(self.c_act(self.conv_start(seq.permute(0,2,1))).view(B, -1))
        # end = self.fc_end(self.c_act(self.conv_end(seq.permute(0,2,1))).view(B, -1))

        start = self.fc_start(seq.reshape(B, -1))
        end = self.fc_end(seq.reshape(B, -1))
        return start, end

# Retrieved from pytorch website


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class SmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0, n_classes=150):
        super(SmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = n_classes
        self.dim = 1

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class Loss(nn.Module):
    def __init__(self, smoothing=0.0, n_classes=150):
        super(Loss, self).__init__()   
        self.start_loss = SmoothingLoss(smoothing=smoothing, n_classes=n_classes)
        self.end_loss = SmoothingLoss(smoothing=smoothing, n_classes=n_classes)

    def forward(self, y_hat_start, start, y_hat_end, end):
        return (self.start_loss(y_hat_start, start) + self.end_loss(y_hat_end, end)) * 0.5
