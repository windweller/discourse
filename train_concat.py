"""
Train a joint model of SNLI and discourse
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import pickle

import tensorflow as tf
import numpy as np

import data
from classifier import SequenceClassifier, Encoder
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("exclude", "", "discourse markers excluded")
tf.app.flags.DEFINE_string("include", "", "discourse markers included")


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def dict_to_list(dic):
    l = [None] * len(dic)
    for k, v in dic.iteritems():
        l[v] = k
    return l


def main(_):
    if not os.path.exists(FLAGS.run_dir):
        os.makedirs(FLAGS.run_dir)
    file_handler = logging.FileHandler("{0}/log.txt".format(FLAGS.run_dir))
    logging.getLogger().addHandler(file_handler)

    if not FLAGS.snli:
        if FLAGS.exclude == "" and FLAGS.include == "":
            tag = "all"
        elif FLAGS.exclude != "":
            tag = "no_" + FLAGS.exclude.replace(",", "_").replace(" ", "_")
            # last part is for "for example"
        elif FLAGS.include != "":
            tag = FLAGS.include.replace(",", "_").replace(" ", "_")
        else:
            raise Exception("no match state for exclude/include")
        glove_name = "glove.trimmed.{}_{}.npz".format(FLAGS.embedding_size, tag)
        vocab_name = "vocab_{}.dat".format(tag)
        tag = "_" + tag
    else:
        logging.info("Training on SNLI")
        tag = ""
        glove_name = "glove.trimmed.300.npz"
        vocab_name = "vocab.dat"

    # now we load in glove based on tags
    embed_path = pjoin(FLAGS.prefix, "data", FLAGS.dataset, glove_name)
    vocab_path = pjoin(FLAGS.prefix, "data", FLAGS.dataset, vocab_name)
    vocab, rev_vocab = initialize_vocab(vocab_path)
    vocab_size = len(vocab)

    logging.info("vocab size: {}".format(vocab_size))

    pkl_train_name = pjoin(FLAGS.prefix, "data", FLAGS.dataset, "train{}.ids.pkl".format(tag))
    pkl_val_name = pjoin(FLAGS.prefix, "data", FLAGS.dataset, "valid{}.ids.pkl".format(tag))
    pkl_test_name = pjoin(FLAGS.prefix, "data", FLAGS.dataset, "test{}.ids.pkl".format(tag))

    with open(pkl_test_name, "rb") as f:
        q_test = pickle.load(f)

    with open(pjoin(FLAGS.prefix, "data", FLAGS.dataset, "class_labels{}.pkl".format(tag)), "rb") as f:
        label_dict = pickle.load(f)
    label_tokens = dict_to_list(label_dict)
    logging.info("classifying markers: {}".format(label_tokens))

    with open(os.path.join(FLAGS.run_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # auto-adjust label size
    discourse_label_size = 14
    if FLAGS.exclude != "":
        discourse_label_size -= len(FLAGS.exclude.split(","))
    elif FLAGS.include != "":
        discourse_label_size = len(FLAGS.include.split(","))
    elif FLAGS.snli:
        discourse_label_size = 3

    # SNLI
    snli_label_size = 3

    with tf.Graph().as_default(), tf.Session() as session:
        tf.set_random_seed(FLAGS.seed)

        # initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale, seed=FLAGS.seed)

        initializer = tf.uniform_unit_scaling_initializer(FLAGS.init_scale, seed=FLAGS.seed)

        with tf.variable_scope("discourse", reuse=None, initializer=initializer):
            encoder = Encoder(size=FLAGS.state_size, num_layers=FLAGS.layers)
            discourse_sc = SequenceClassifier(encoder, FLAGS, vocab_size, vocab, rev_vocab, discourse_label_size, embed_path,
                                    optimizer=FLAGS.opt)

        with tf.variable_scope("snli", reuse=None, initializer=initializer):
            # preparation for SNLI

            snli_glove_name = "glove.trimmed.300.npz"
            snli_embed_path = pjoin(FLAGS.prefix, "data", "snli", snli_glove_name)

            vocab_name = "vocab.dat"

            vocab_path = pjoin(FLAGS.prefix, "data", "snli", vocab_name)
            vocab, rev_vocab = initialize_vocab(vocab_path)
            vocab_size = len(vocab)

            logging.info("SNLI vocab size: {}".format(vocab_size))

            snli_encoder = Encoder(size=FLAGS.state_size, num_layers=FLAGS.layers)
            snli_sc = SequenceClassifier(snli_encoder, FLAGS, vocab_size, vocab, rev_vocab, snli_label_size, snli_embed_path,
                                    optimizer=FLAGS.opt)

        model_saver = tf.train.Saver(max_to_keep=FLAGS.epochs)

        if FLAGS.restore_checkpoint is not None:
            model_saver.restore(session, FLAGS.restore_checkpoint)
            logging.info("model loaded")
        else:
            tf.global_variables_initializer().run()

        if not FLAGS.dev:
            # restore_epoch by default is 0
            with open(pkl_train_name, "rb") as f:
                q_train = pickle.load(f)

            with open(pkl_val_name, "rb") as f:
                q_valid = pickle.load(f)

            discourse_sc.but_because_train(session, q_train, q_valid, q_test, label_tokens, FLAGS.restore_epoch, FLAGS.epochs, FLAGS.run_dir)
        else:
            discourse_sc.but_because_dev_test(session, q_test, FLAGS.run_dir, label_tokens)

        logging.info("entering SNLI training")

        if not FLAGS.dev:
            # load in SNLI files (change this when we load in Multi-NLI)
            pkl_train_name = pjoin(FLAGS.prefix, "data", "snli", "train.ids.pkl")
            pkl_val_name = pjoin(FLAGS.prefix, "data", "snli", "valid.ids.pkl")
            pkl_test_name = pjoin(FLAGS.prefix, "data", "snli", "test.ids.pkl")

            with open(pkl_train_name, "rb") as f:
                q_train = pickle.load(f)

            with open(pkl_val_name, "rb") as f:
                q_valid = pickle.load(f)

            with open(pkl_test_name, "rb") as f:
                q_test = pickle.load(f)

            snli_sc.but_because_train(session, q_train, q_valid, q_test, label_tokens, FLAGS.restore_epoch, FLAGS.epochs, FLAGS.run_dir)

if __name__ == "__main__":
    tf.app.run()
