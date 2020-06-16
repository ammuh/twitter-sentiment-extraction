import torch
from data import Tweets
from model import Embed, TransformerSentiment
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

device = 'cuda:0'
dataset = Tweets(device, test=True)


V = len(dataset.vocab.keys())
P = len(dataset.pos_set.keys())
model = TransformerSentiment(V, P, 132, 132, 12, nhead=12, dropout=.2).to(device)

model.load_state_dict(torch.load('./models/1592284148.pth'))

model.eval()
file1 = open("submission.csv", "w")
file1.write('textID,selected_text\n')

dataloader = DataLoader(dataset, batch_size=100,
                        num_workers=12, pin_memory=True)

with torch.no_grad():

    model.eval()
    with torch.no_grad():

        pbar = tqdm(desc='Submission', total=len(dataloader), leave=False)

        for i, batch in enumerate(dataloader):
            tid = batch['tid']
            tweet = batch['tweet'].to(device)
            sentiment = batch['sentiment']
            offsets = batch['offsets']
            raw_tweet = batch['raw_tweet']
            pos = batch['pos'].to(device)

            y_hat_start, y_hat_end = model(tweet, pos)

            y_hat_start = torch.argmax(y_hat_start, dim=1)
            y_hat_end = torch.argmax(y_hat_end, dim=1)

            final = []

            for j, t in enumerate(tweet):
                s = offsets[j][y_hat_start[j]][0]
                e = offsets[j][y_hat_end[j]][1]

                file1.write('{},"{}"\n'.format(tid[j], raw_tweet[j][s:e]))

            pbar.update()
        pbar.clear()
        pbar.close()

file1.close()
