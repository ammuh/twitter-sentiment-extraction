# Twitter Sentiment Extraction

Submission to https://www.kaggle.com/c/tweet-sentiment-extraction/.
## Dataset
This kaggle dataset is a set of ~27,000 raw text tweets, tagged with sentiments "positive", "negative", or neutral. The goal is to extract a substring from the tweet that represents the support for the given sentiment. For example:
[*This challenge is really cool*,  **positive**]  would have the output of *cool*.
## Approach
Majority of solutions seem to go with pre-trained language models which were fine-tuned for this task specifically. This repository tries to explore training from scratch and enhancing performance by careful tokenization and data augmentation.  I define the dataset as:

![enter image description here](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BD%7D%20%3D%20%5Cleft%20%5C%7B%20%5C%28x%2C%20y_%7Bstart%7D%2C%20y_%7Bend%7D%5C%29_i%5Cright%20%5C%7D%5EN_%7Bi%3D1%7D%20%5C%5C)

x represents the sequence of tokens, y<sub>start</sub> is the index of the starting token, and  y<sub>end</sub> is the index of the ending tokens. The indices are predicted via classification as opposed to regression. The sequence will be modeled with a common representation, and then be predicted with a single linear layer for each output (start and end).

![](https://latex.codecogs.com/gif.latex?%5C%5C%20%5Chat%7By%7D_%7Bstart%7D%28x%29%20%3D%20f%28%5CGamma%28x%29%29%20%5C%5C%20%5Chat%7By%7D_%7Bend%7D%28x%29%20%3D%20g%28%5CGamma%28x%29%29)

*f* and *g* represent the linear layers for each respective output, and **Γ** is the sequence representation. 

Loss is computed as cross entropy loss for the two tasks separately, and then added together. 

## Architecture
The following describes the architecture and training process.

1. **Tokenization:** Raw tweets are tokenized using BertWordPiece tokenization. Given the casual language style of twitter, it's important to train at the subword level. Here I used the BERT uncased base token vocabulary. Sequences are padded to a fixed length.
2. **Part Of Speech**: Using the nltk PennTreebank POS tagger, I determine the part of speech of the word if possible. This will help give context on whether a word can help support sentiment or not.
3. **Data Preparation**: The original BERT vocabulary has three special tokens added, for each sentiment. These tokens are appended to inject sentiment information directly in the sequence. Start and end indexes are calculated, and then all wrapped in a dataset and dataloader.
4. **Model**
    1. **Embeddings**: Each token has a trainable embedding, which is added to a fixed positional encoding, and a trainable positional encoding.
    2. **Transformer Encoder**: To model the sequence, we have a transformer encoder module, which we can specify hidden size, number of multi-attention heads, and layers. This will allow the model to learn bidirectional dependencies for the data representation.
    3. **Task Linear Layer**: Two separate linear layers that output the start and end index of the substring.
5. **Loss**: Cross Entropy loss is computed on each output, and then averaged. 
6. **Optimization**: The model is optimized using an Adam optimizer.

## Generalization
The biggest challenge of this dataset is the small size. Here are the methods to improve performance:

1. **Data Augmentation**: To generate more data, there are two augmentation operations performed on the original data. First words are randomly inserted using contextualized embeddings from BERT embeddings. Then words are randomly replaced with their synonyms from nltk wordnet synsets.
2. **Label Smoothing** : Cross entropy loss is adjusted to have smoothing loss.
