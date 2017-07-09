#! /usr/bin/env python

"""
Version of class paper in PyTorch
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from os.path import isfile
from os.path import join as pjoin

import numpy as np

from util_pytorch import pair_iter

# =================== set parameters ===================

data_dir = pjoin("..", "data", "ptb")
batch_size = 10 #100
max_seq_len = 35

embedding_size = 100

hidden_size = 5 #256
output_size = 2
n_layers = 1
n_epochs = 2 #8

np.random.seed(123)

# =================== initialize vocab ===================

def initialize_vocab(vocab_path):
    if isfile(vocab_path):
        rev_vocab = []
        with open(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

vocab_path = pjoin(data_dir, "vocab.dat")
vocab, rev_vocab = initialize_vocab(vocab_path)

# =================== load data ===================

example_generator = pair_iter(task="but_because", data_dir=data_dir,
	                          split="train", vocab=vocab, rev_vocab=rev_vocab,
	                          batch_size=batch_size, max_seq_len=max_seq_len,
	                          shuffle=True, cache=False)

s1_tokens, s1_mask, s2_tokens, s2_mask, y, text = next(example_generator)

# =================== convert to input format ===================

glove_path = pjoin(data_dir, "glove.trimmed.{}.npz".format(embedding_size))
glove_embeddings = np.load(glove_path)["glove"]

# seq_len, batch_size, nfeatures
batch = np.array([glove_embeddings[i] for i in s1_tokens.transpose()])

# check that dimensions are (probably) correct
assert(batch.shape[0] <= max_seq_len)
assert(batch.shape[1] == batch_size)
assert(batch.shape[2] == embedding_size)

batch_tensor = torch.Tensor(batch)

# =================== convert to output to one-hot ===================

# =================== initialize model ===================

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            bidirectional=True)

        self.W = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden, cell):
        sequence_output, last_layer = self.lstm(input, (hidden, cell))
        hidden, cell = last_layer
        fT = hidden[0]
        bT = hidden[1]
        hT = torch.cat((fT, bT), 0)
        y = self.softmax(self.W(hT))
        return y

    def initHidden(self):
        return Variable(torch.zeros(2, batch_size, hidden_size))

model = Net()

model_input = Variable(batch_tensor)
hidden = model.initHidden()
cell = model.initHidden()

print(model(model_input, hidden, cell))

# # define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# self.logits = rnn_cell._linear([seqA_c_vec, seqB_c_vec, persA_B_mul, persA_B_sub, persA_B_avg],
#                                        self.label_size, bias=True)
# self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.labels))
