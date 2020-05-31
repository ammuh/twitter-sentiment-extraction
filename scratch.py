# from tokenizers import ByteLevelBPETokenizer, Tokenizer, models, pre_tokenizers, decoders, trainers, processors
# import os

# tokenizer = None

# if not os.path.isfile("{}-vocab.json".format('token_test')):

#     # Initialize a tokenizer
#     tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)

#     # And then train
#     tokenizer.train(
#          ["./data/train_corpus.txt"],
#         vocab_size=10000,
#         min_frequency=2,
#         show_progress=True,
#     )


#     print(tokenizer)
#     tokenizer.save('./', 'token_test')
# else:

#     tokenizer = ByteLevelBPETokenizer( "./{}-vocab.json".format('token_test'), "./{}-merges.txt".format('token_test'),
#         add_prefix_space=True,
#     )

# # Now we can encode
# encoded = tokenizer.encode("will be back later.  http://plurk.com/p/rp3k7,will be back later, loooove u @mahboi #blessed")
# print(encoded.tokens)
# print(encoded.offsets)

from tokenizers import BertWordPieceTokenizer
# My arbitrary sentence
sentence = "[CLS] will be back later.  www.facebook.com ,will be back later, loooove u @mahboi #blessed"
# Bert vocabularies
# Instantiate a Bert tokenizers
tokenizer = BertWordPieceTokenizer("bert-large-uncased-vocab.txt", lowercase=True, clean_text=True)
tokenizer.add_tokens(['[LINK]'])

tokenizer.enable_padding(max_length=100)
WordPieceEncoder = tokenizer.encode(sentence)
# Print the ids, tokens and offsets
print(WordPieceEncoder.ids)
print(WordPieceEncoder.tokens)
print(WordPieceEncoder.offsets)
print(tokenizer.get_vocab()['[PAD]'])
print(tokenizer.decode(WordPieceEncoder.ids))