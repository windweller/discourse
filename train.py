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

    embed_path = FLAGS.embed_path or pjoin("data", FLAGS.dataset, "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = pjoin("data", FLAGS.dataset, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)
    vocab_size = len(vocab)

    logging.info("vocab size: {}".format(vocab_size))

    if FLAGS.exclude == "" and FLAGS.include == "":
        tag = "all"
    elif FLAGS.exclude != "":
        tag = "no_" + FLAGS.exclude.replace(",", "_").replace(" ", "_")
        # last part is for "for example"
    elif FLAGS.include != "":
        tag = FLAGS.include.replace(",", "_").replace(" ", "_")
    else:
        raise Exception("no match state for exclude/include")

    pkl_train_name = pjoin("data", FLAGS.dataset, "train_{}.ids.pkl".format(tag))
    pkl_val_name = pjoin("data", FLAGS.dataset, "valid_{}.ids.pkl".format(tag))
    pkl_test_name = pjoin("data", FLAGS.dataset, "test_{}.ids.pkl".format(tag))

    with open(pkl_train_name, "rb") as f:
        q_train = pickle.load(f)

    with open(pkl_val_name, "rb") as f:
        q_valid = pickle.load(f)

    with open(pkl_test_name, "rb") as f:
        q_test = pickle.load(f)

    with open(pjoin("data", FLAGS.dataset, "class_labels_{}.pkl".format(tag)), "rb") as f:
        label_dict = pickle.load(f)
    label_tokens = dict_to_list(label_dict)
    logging.info("classifying markers: {}".format(label_tokens))

    data_dir = pjoin("data", FLAGS.dataset)

    with open(os.path.join(FLAGS.run_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # auto-adjust label size
    label_size = 14
    if FLAGS.exclude != "":
        label_size -= len(FLAGS.exclude.split(","))
    elif FLAGS.include != "":
        label_size = len(FLAGS.include.split(","))

    with tf.Graph().as_default(), tf.Session() as session:
        tf.set_random_seed(FLAGS.seed)

        # initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale, seed=FLAGS.seed)

        initializer = tf.uniform_unit_scaling_initializer(FLAGS.init_scale, seed=FLAGS.seed)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            encoder = Encoder(size=FLAGS.state_size, num_layers=FLAGS.layers)
            sc = SequenceClassifier(encoder, FLAGS, vocab_size, vocab, rev_vocab, label_size, embed_path,
                                    optimizer=FLAGS.opt)

        model_saver = tf.train.Saver(max_to_keep=FLAGS.epochs)

        if FLAGS.restore_checkpoint is not None:
            model_saver.restore(session, FLAGS.restore_checkpoint)

        if not FLAGS.dev:
            tf.global_variables_initializer().run()
            sc.but_because_train(session, q_train, q_valid, q_test, label_tokens, 0, FLAGS.epochs, FLAGS.run_dir)

        else:
            sc.but_because_dev_test(session, data_dir, FLAGS.run_dir, FLAGS.best_epoch, label_tokens)

if __name__ == "__main__":
    tf.app.run()
