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

from six.moves import urllib

from tensorflow.python.platform import gfile
from tqdm import *
import numpy as np
from os.path import join as pjoin

_PAD = b"<pad>" # no need to pad
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1

"""
# preprossesing for seq to seq (data_gen)
# data reads preprocessed files

* make vocab files
    - vocab
        dict {word: number}
    - reverse vocab
        list[number] = word
* download and load glove
    - 6b
    - 300d

save vocab into file vocab.dat
just a text file where each line is a word type

function naming convention

train_BECAUSE
tokenized version of data: train_BECAUSE.ids.txt
this gets passed into pair_iter...
"""

def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    vocab_dir = os.path.join("data", "winograd")
    glove_dir = os.path.join("data", "glove.6B")
    source_dir = os.path.join("data", "winograd")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=100, type=int)
    parser.add_argument("--random_init", action='store_true')
    return parser.parse_args()


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
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

"""
once we have vocabulary, go through and make trimmed down word matrix

convert indices in my vocabulary to indices in glove
not all words will overlap in both vocabs.

trained glove will be somewhere in the directory

(last quarter this was wrong, but this is probably correct. maybe check git history)
cs224n website has this. pa4 code.
"""
def process_glove(args, vocab_list, save_path, size=4e5, random_init=True):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if gfile.Exists(save_path + ".npz"):
        print("Glove file already exists at %s" % (save_path + ".npz"))
    else:
        glove_path = os.path.join(args.glove_dir, "glove.6B.{}d.txt".format(args.glove_dim))
        if random_init:
            glove = np.random.randn(len(vocab_list), args.glove_dim)
        else:
            glove = np.zeros((len(vocab_list), args.glove_dim))
        found = 0
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh, total=size):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                    found += 1
                if word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                    found += 1
                if word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                    found += 1
                if word.lower() in vocab_list:
                    idx = vocab_list.index(word.lower())
                    glove[idx, :] = vector
                    found += 1

        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))

def create_vocabulary(vocabulary_path, data_paths, tokenizer=None):
    if gfile.Exists(vocabulary_path):
        print("Vocabulary file already exists at %s" % vocabulary_path)
    else:
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = {}
        for path in data_paths:
            with open(path, mode="rb") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                    for w in tokens:
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

"""
tokenize and map words to ids in vocab
called by data_to_token_ids
"""
def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 5000 == 0:
                        print("tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

if __name__ == '__main__':
    args = setup_args()

    vocab_path = pjoin(args.vocab_dir, "vocab.dat")

    partial_fnames = ["test_BECAUSE", "test_BUT",
                      "train_BECAUSE", "train_BUT",
                      "valid_BECAUSE", "valid_BUT"]

    data_fnames = [partial_fname + ".txt" for partial_fname in partial_fnames]
    data_paths = [pjoin(args.source_dir, fname) for fname in data_fnames]

    create_vocabulary(vocab_path, data_paths, tokenizer=None) # nltk.word_tokenize

    vocab, rev_vocab = initialize_vocabulary(pjoin(args.vocab_dir, "vocab.dat"))

    # ======== Trim Distributed Word Representation =======
    # If you use other word representations, you should change the code below

    process_glove(args, rev_vocab, args.source_dir + "/glove.trimmed.{}".format(args.glove_dim),
                  random_init=args.random_init)

    # ======== Creating Dataset =========
    # We created our data files seperately
    # If your model loads data differently (like in bulk)
    # You should change the below code

    for partial_fname in partial_fnames:
        data_path = pjoin(args.source_dir, partial_fname + ".txt")
        ids_path = pjoin(args.source_dir, partial_fname + ".ids.txt")
        data_to_token_ids(data_path, ids_path, vocab_path, tokenizer=None) # nltk.word_tokenize
