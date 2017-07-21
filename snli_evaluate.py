"""
We load in the sentence representation
that is pre-trained and we evaluate on different
tasks including:
1. SNLI
2. SST
"""

from torchtext import data
from torchtext import datasets

from os.path import join as pjoin

import tensorflow as tf
from classifier import SequenceClassifier, Encoder

import logging
logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    inputs = data.Field(lower=True)
    answers = data.Field(sequential=False)

    train, dev, test = datasets.SNLI.splits(inputs, answers, root=pjoin("data", "snli"))

    inputs.build_vocab(train, dev, test)

    # TODO: Consider rewrite the Vocab class into TF compatible class
    # TODO: fully leverage the pre-processing and GloVE trimming

    inputs.vocab.load_vectors()

    answers.build_vocab(train)

    # we build the model and load the parameters
    embed_path = FLAGS.embed_path or pjoin("data", "snli", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
