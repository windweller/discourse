from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json

import tensorflow as tf
import numpy as np

from classifier import SequenceClassifier, Encoder
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

# TODO: copy this file and make one for cause_effect

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


def main(_):
    if not os.path.exists(FLAGS.run_dir):
        os.makedirs(FLAGS.run_dir)
    file_handler = logging.FileHandler("{0}/log.txt".format(FLAGS.run_dir))
    logging.getLogger().addHandler(file_handler)

    embed_path = FLAGS.embed_path or pjoin("data", FLAGS.dataset, "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = pjoin("data", FLAGS.dataset, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)
    vocab_size = len(vocab)

    but_train = pjoin("data", FLAGS.dataset, "train_BUT.ids.txt")
    because_train = pjoin("data", FLAGS.dataset, "train_BECAUSE.ids.txt")

    but_valid = pjoin("data", FLAGS.dataset, "valid_BUT.ids.txt")
    because_valid = pjoin("data", FLAGS.dataset, "valid_BECAUSE.ids.txt")

    # in dev setting, these would be dev
    but_test = pjoin("data", FLAGS.dataset, "test_BUT.ids.txt")
    because_test = pjoin("data", FLAGS.dataset, "test_BECAUSE.ids.txt")

    data_dir = pjoin("data", FLAGS.dataset)

    with open(os.path.join(FLAGS.run_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Graph().as_default(), tf.Session() as session:
        tf.set_random_seed(FLAGS.seed)

        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale, seed=FLAGS.seed)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            encoder = Encoder(size=FLAGS.state_size, num_layers=FLAGS.layers)
            sc = SequenceClassifier(encoder, FLAGS, vocab_size, vocab, rev_vocab, embed_path, task=FLAGS.task)

        model_saver = tf.train.Saver(max_to_keep=FLAGS.epochs)

        if FLAGS.restore_checkpoint is not None:
            model_saver.restore(session, FLAGS.restore_checkpoint)

        if not FLAGS.dev:
            tf.global_variables_initializer().run()
            if FLAGS.task == "but":
                sc.but_because_train(session, but_train, because_train, but_valid,
                                             because_valid, but_test, because_test,
                                             0, FLAGS.epochs, FLAGS.run_dir, data_dir)
            else:
                sc.cause_effect_train(session, data_dir, because_train, because_valid,because_test,
                                                   0, FLAGS.epochs, FLAGS.run_dir)
        else:
            if FLAGS.task == "but":
                sc.but_because_dev_test(session, data_dir, FLAGS.run_dir, FLAGS.best_epoch)
            else:
                sc.cause_effect_dev_test(session, data_dir, because_valid, FLAGS.run_dir, FLAGS.best_epoch)

if __name__ == "__main__":
    tf.app.run()
