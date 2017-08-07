from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile
import argparse
import nltk
import string
import pickle
import random

from six.moves import urllib

from tensorflow.python.platform import gfile
from tqdm import *
import numpy as np
from os.path import join as pjoin

_PAD = b"<pad>"  # no need to pad
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1

import sys

reload(sys)
sys.setdefaultencoding('utf8')

np.random.seed(123)
random.seed(123)


def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    vocab_dir = os.path.join("data", "wikitext-103")
    glove_dir = os.path.join("data", "glove.6B")
    source_dir = os.path.join("data", "wikitext-103")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=300, type=int)
    parser.add_argument("--random_init", action='store_true')
    parser.add_argument("--train_size", default=0.9)
    parser.add_argument("--undersamp_cutoff", default=50000, type=int)


def basic_tokenizer(sentence):
    words = []
    # this is stripping punctuations
    for space_separated_fragment in ("".join(c for c in sentence if c not in string.punctuation)).split():
        words.extend(re.split(" ", space_separated_fragment))

    return [w for w in words if w]


def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def process_glove(args, vocab_dict, save_path, random_init=True):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if gfile.Exists(save_path + ".npz"):
        print("Glove file already exists at %s" % (save_path + ".npz"))
    else:
        glove_path = os.path.join(args.glove_dir, "glove.840B.{}d.txt".format(args.glove_dim))
        if random_init:
            glove = np.random.randn(len(vocab_dict), args.glove_dim)
        else:
            glove = np.zeros((len(vocab_dict), args.glove_dim))

        found = 0

        with open(glove_path, 'r') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in vocab_dict:  # all cased
                    idx = vocab_dict[word]
                    glove[idx, :] = np.fromstring(vec, sep=' ')
                    found += 1

        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


def create_vocabulary(vocabulary_path, sentence_pairs_data, discourse_markers=None):
    if gfile.Exists(vocabulary_path):
        print("Vocabulary file already exists at %s" % vocabulary_path)
    else:
        print("Creating vocabulary {}".format(vocabulary_path))
        vocab = {}
        counter = 0
        if not discourse_markers:
            discourse_markers = sentence_pairs_data.keys()
        for discourse_marker in discourse_markers:
            for s1, s2 in sentence_pairs_data[discourse_marker]:
                counter += 1
                if counter % 100000 == 0:
                    print("processing line %d" % counter)
                for w in s1:
                    if not w in _START_VOCAB:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
                for w in s2:
                    if not w in _START_VOCAB:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")

def sentence_to_token_ids(sentence, vocabulary):
    return [vocabulary.get(w, UNK_ID) for w in sentence]

