from __future__ import absolute_import, division, print_function
from copy import deepcopy
import time
import os
import logging

import tensorflow as tf
import numpy as np

from os.path import join as pjoin

flags = tf.flags

flags.DEFINE_integer("hidden_dim", 512, "hidden dimension")
flags.DEFINE_integer("layers", 2, "number of hidden layers")
flags.DEFINE_integer("unroll", 35, "number of time steps to unroll for BPTT, also the max sequence length")
flags.DEFINE_integer("batch_size", 20, "batch size")
flags.DEFINE_float("init_scale", 0.1, "scale for random initialization")
flags.DEFINE_float("learning_rate", 1.0, "initial learning rate")
flags.DEFINE_float("learning_rate_decay", 0.5, "amount to decrease learning rate")

logging.basicConfig(level=logging.INFO)


class SequenceClassifier(object):
    def __init__(self, flags, vocab_size, embed_path, task, is_training=True):
        # task: ["but", "cause"]

        batch_size = flags.batch_size
        unroll = flags.unroll
        self.embed_path = embed_path
        self.vocab_size = vocab_size
        self.flags = flags

        self.seqA = tf.placeholder(tf.int32, [batch_size, unroll])
        self.seqB = tf.placeholder(tf.int32, [batch_size, unroll])
        self.seqA_mask = tf.placeholder(tf.int32, [batch_size, unroll])
        self.seqB_mask = tf.placeholder(tf.int32, [batch_size, unroll])

        with tf.device("/cpu:0"):
            embed = tf.constant(np.load(self.embed_path)['glove'], dtype=tf.float32, name="glove",
                                shape=[self.vocab_size, self.flags.embedding_size])
            self.seqA_inputs = tf.nn.embedding_lookup(embed, self.seqA)
            self.seqB_inputs = tf.nn.embedding_lookup(embed, self.seqB)

        # main computation graph is here

    def setup_but_because(self):
        # seqA: but, seqB: because, this will learn to differentiate them
        pass

    def setup_cause_effect(self):
        pass

