from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import pickle

import numpy as np
import argparse

from os.path import join as pjoin

import logging

np.random.seed(123)

"""
python train.py --dataset books --include "but,because,when,if,for example,so,before,still" --run_dir int_simple8
"""

# from classifier import SequenceClassifier, Encoder

logging.basicConfig(level=logging.INFO)

def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    glove_dir = os.path.join("../data", "glove.6B")

    #discourse markers excluded
    parser.add_argument("--exclude", default="", type=str)

    #discourse markers included
    parser.add_argument("--include", default="", type=str)

    #directory to store experiment outputs
    parser.add_argument("--run_dir", default="sandbox", type=str)

    #if flag True, the classifier will train on SNLI
    parser.add_argument("--snli", action='store_true')

    #ptb/wikitext-103/books select the dataset to use
    parser.add_argument("--dataset", default="wikitext-103", type=str)

    #dimension of GloVE vector to use
    parser.add_argument("--embedding_size", default=300, type=int)

    return parser.parse_args()


def initialize_vocab(vocab_path):
    if os.path.exists(vocab_path):
        rev_vocab = []
        with open(vocab_path, mode="rb") as f:
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


if __name__ == "__main__":
    args = setup_args()
    print(vars(args))

    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)
    file_handler = logging.FileHandler("{0}/log.txt".format(args.run_dir))
    logging.getLogger().addHandler(file_handler)

    if not args.snli:
        if args.exclude == "" and args.include == "":
            tag = "all"
        elif args.exclude != "":
            tag = "no_" + args.exclude.replace(",", "_").replace(" ", "_")
            # last part is for "for example"
        elif args.include != "":
            tag = args.include.replace(",", "_").replace(" ", "_")
        else:
            raise Exception("no match state for exclude/include")
        glove_name = "glove.trimmed.{}_{}.npz".format(args.embedding_size, tag)
        vocab_name = "vocab_{}.dat".format(tag)
        tag = "_" + tag
    else:
        logging.info("Training on SNLI")
        tag = "snli"
        glove_name = "glove.trimmed.300.npz"
        vocab_name = "vocab.dat"

    # now we load in glove based on tags
    embed_path = pjoin("../data", args.dataset, glove_name)
    vocab_path = pjoin("../data", args.dataset, vocab_name)
    vocab, rev_vocab = initialize_vocab(vocab_path)
    vocab_size = len(vocab)

    logging.info("vocab size: {}".format(vocab_size))

    pkl_train_name = pjoin("../data", args.dataset, "train{}.ids.pkl".format(tag))
    pkl_val_name = pjoin("../data", args.dataset, "valid{}.ids.pkl".format(tag))
    pkl_test_name = pjoin("../data", args.dataset, "test{}.ids.pkl".format(tag))

    with open(pkl_train_name, "rb") as f:
        q_train = pickle.load(f)

    with open(pkl_val_name, "rb") as f:
        q_valid = pickle.load(f)

    with open(pkl_test_name, "rb") as f:
        q_test = pickle.load(f)

    with open(pjoin("../data", args.dataset, "class_labels{}.pkl".format(tag)), "rb") as f:
        label_dict = pickle.load(f)
    label_tokens = dict_to_list(label_dict)
    logging.info("classifying markers: {}".format(label_tokens))

    data_dir = pjoin("../data", args.dataset)

    with open(os.path.join(args.run_dir, "args.json"), 'w') as fout:
        json.dump(vars(args), fout)

    # auto-adjust label size
    label_size = 14
    if args.exclude != "":
        label_size -= len(args.exclude.split(","))
    elif args.include != "":
        label_size = len(args.include.split(","))
    elif args.snli:
        label_size = 3

    # with tf.Graph().as_default(), tf.Session() as session:
    #     tf.set_random_seed(FLAGS.seed)

    #     # initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale, seed=FLAGS.seed)

    #     initializer = tf.uniform_unit_scaling_initializer(FLAGS.init_scale, seed=FLAGS.seed)

    #     with tf.variable_scope("model", reuse=None, initializer=initializer):
    #         encoder = Encoder(size=FLAGS.state_size, num_layers=FLAGS.layers)
    #         sc = SequenceClassifier(encoder, FLAGS, vocab_size, vocab, rev_vocab, label_size, embed_path,
    #                                 optimizer=FLAGS.opt)

    #     model_saver = tf.train.Saver(max_to_keep=FLAGS.epochs)

    #     if FLAGS.restore_checkpoint is not None:
    #         model_saver.restore(session, FLAGS.restore_checkpoint)
    #         logging.info("model loaded")
    #     else:
    #         tf.global_variables_initializer().run()

    #     if not FLAGS.dev:
    #         # restore_epoch by default is 0
    #         sc.but_because_train(session, q_train, q_valid, q_test, label_tokens, FLAGS.restore_epoch, FLAGS.epochs, args.run_dir)
    #     else:
    #         sc.but_because_dev_test(session, q_test, args.run_dir, label_tokens)
