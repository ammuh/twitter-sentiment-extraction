import torch
from data import Tweets
from model import Embed, LSTM, LSTMSentiment, TransformerSentiment

data = Tweets('cuda:0', test=True)


V = len(data.vocab.keys())

model = LSTMSentiment(V, 64, 64, 2).to('cuda:0')

model.load_state_dict(torch.load('blstm_64_64_2_fold.pth'))


file1 = open("submission.csv","w") 
file1.write('textID,selected_text\n')
for row in data:
    tid, tweet, sentiments = tuple(row)
    

    
    y_hat_full = model(tweet.unsqueeze(0), sentiments.unsqueeze(0))#.reshape(len(batch), dataset.pad, -1)


    y_hat = torch.argmax(y_hat_full[0], dim=1)

    selection_output = data.tokenizer.decode(list(y_hat * tweet))
    print(selection_output)
    file1.write('{},{}\n'.format(tid, selection_output))

file1.close()