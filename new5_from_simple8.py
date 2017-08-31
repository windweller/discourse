#! /usr/bin/env python
# -*- coding: utf-8 -*-

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

_PAD = b"<pad>" # no need to pad
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1

import sys
reload(sys)
sys.setdefaultencoding('utf8')

np.random.seed(123)
random.seed(123)

"""
Loads data in format:
```
{
    "discourse marker": [
        (["tokens", "in", "s1"], ["tokens", "in", "s2"]),
        ...
    ],
    ...
}
```

Exports data (split into valid, train, test files) in format:
```
[(s1, s2, label), ...]
```
where `label` is class index from {0, ..., number_of_discourse_markers},
and `s1` and `s2` are lists of word ids

Also creates vocabulary file `vocab.dat` specifying the mapping between glove embeddings and ids.

python data.py --dataset books --data_name all_pairs_all_markers_clean_ssplit.pkl --data_tag new5 --include "but,because,when,if,for example,so,before,still" --undersamp_cutoff

"""


def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    glove_dir = os.path.join("data", "glove.6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_dim", default=300, type=int)
    parser.add_argument("--random_init", action='store_true')
    return parser.parse_args()

    # train.pkl -> train_no_because.pkl, train.pkl -> train_all.pkl

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

"""
once we have vocabulary, go through and make trimmed down word matrix

convert indices in my vocabulary to indices in glove
not all words will overlap in both vocabs.

trained glove will be somewhere in the directory

(last quarter this was wrong, but this is probably correct. maybe check git history)
cs224n website has this. pa4 code.
"""
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
            for s1, s2, label in sentence_pairs_data[discourse_marker]:
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

"""
map words to ids in vocab
called by data_to_token_ids
"""
def sentence_to_token_ids(sentence, vocabulary):
    return [vocabulary.get(w, UNK_ID) for w in sentence]


def data_to_token_ids(data, rev_class_labels, target_path, text_path, vocabulary_path, data_dir):
    if gfile.Exists(target_path):
        print("file {} already exists".format(target_path))
    else:
        vocab, _ = initialize_vocabulary(vocabulary_path)

        # fix me: this will be a list instead
        ids_data = []
        text_data = []
        for marker in data:
            # ids_data[marker] = []
            counter = 0
            for s1, s2, label in data[marker]:
                counter += 1
                if counter % 10000 == 0:
                    print("converting %s %d" % (marker, counter))
                token_ids_s1 = sentence_to_token_ids(s1, vocab)
                token_ids_s2 = sentence_to_token_ids(s2, vocab)
                ids_data.append((token_ids_s1, token_ids_s2, label))
                text_data.append((s1, s2, label))

        shuffled_idx = range(len(ids_data))
        random.shuffle(shuffled_idx)
        shuffled_ids_data = [ids_data[idx] for idx in shuffled_idx]
        shuffled_text_data = [text_data[idx] for idx in shuffled_idx]

        print("writing {} and {}".format(target_path, text_path))
        pickle.dump(shuffled_ids_data, gfile.GFile(target_path, mode="wb"))

        # with gfile.GFile(text_path, mode="wb") as f:
        #     for t in shuffled_text_data:
        #         f.write(str([" ".join(t[0]), " ".join(t[1]), rev_class_labels[t[2]]]) + "\n")


def tokenize_sentence_pair_data(sentence_pairs_data):
    tokenized_sent_pair_data = {}
    for key, value in sentence_pairs_data.iteritems():
        sent_pairs = []
        for sent_pair in value:
            sent_pairs.append((sent_pair[0].split(), sent_pair[1].split()))
        tokenized_sent_pair_data[key] = sent_pairs

    return tokenized_sent_pair_data


def extract_new5(split, source_dir, new5_markers, rev_vocab, rev_labels):
    old_filename = pjoin(
        source_dir,
        split + "_but_because_when_if_for_example_so_before_still.ids.pkl"
    )
    old_data = pickle.load(open(old_filename, "rb"))
    raw_data = {marker: [] for marker in new5_markers}
    for s1, s2, label in old_data:
        marker = rev_labels[label]
        if marker in new5_markers:
            words1 = [rev_vocab[i] for i in s1]
            words2 = [rev_vocab[i] for i in s2]
            raw_data[marker].append((words1, words2, label))
    return raw_data

def redo_vocab(valid, train, test, args, new5_markers, new5_tag, vocab_dir):
    all_pairs = {}
    for marker in new5_markers:
        all_pairs[marker] = valid[marker] + test[marker] + train[marker]

    new_vocab_path = pjoin(vocab_dir, "vocab_but_because_when_if_so.dat")
    create_vocabulary(new_vocab_path, all_pairs)

"""
 - sampling procedure (don't spend too much time on this!)
       - pkl format [[s_1, s_2, label]]
            * label is class index {0, ..., N}
       - data split and shuffling: train, valid, test
       - create_vocabulary as in notebook
       - list_to_indices
       - np.random.shuffle FIX RANDOM SEED
       - create_dataset
             * does the split, writes the files!
"""
if __name__ == '__main__':
    args = setup_args()

    vocab_dir = os.path.join("data", "books")
    source_dir = os.path.join("data", "books")

    class_labels = pickle.load(
        open(
            pjoin(source_dir, "class_labels_but_because_when_if_for_example_so_before_still.pkl"), 
            "rb"
        )
    )
    rev_labels = [marker for marker in class_labels]
    for marker in class_labels:
        label = class_labels[marker]
        rev_labels[label] = marker

    new5_markers = ["but", "because", "when", "if", "so"]
    new5_tag = "_".join(new5_markers)

    vocab, rev_vocab = initialize_vocabulary(
        pjoin(vocab_dir, "vocab_but_because_when_if_for_example_so_before_still.dat")
    )

    print("loading valid")
    valid = extract_new5("valid", source_dir, new5_markers, rev_vocab, rev_labels)
    print("loading train")
    train = extract_new5("train", source_dir, new5_markers, rev_vocab, rev_labels)
    print("loading test")
    test = extract_new5("test", source_dir, new5_markers, rev_vocab, rev_labels)


    new_vocab_path = pjoin(vocab_dir, "vocab_but_because_when_if_so.dat")
    if not gfile.Exists(new_vocab_path):
        print("redoing vocab")
        redo_vocab(valid, train, test, args, new5_markers, new5_tag, vocab_dir)

    vocab, rev_vocab = initialize_vocabulary(new_vocab_path)

    print("glove")
    process_glove(
        args, 
        vocab, 
        "glove.trimmed.{}_{}.npz".format(args.glove_dim, new5_tag),
        random_init=args.random_init
    )

    print("combining data")
    splits = {"valid": valid, "train": train, "test": test}

    class_labels = {}
    rev_class_labels = []
    i=0
    for marker in new5_markers:
        i += 1
        class_labels[marker] = i
        rev_class_labels.append(marker)
    
    for split in ["valid", "train", "test"]:
        print("Converting data in {}".format(split))

        data = splits[split]
        ids_path = pjoin(
            source_dir,
            "{}_{}.ids.pkl".format(split, new5_tag)
        )
        text_path = pjoin(
            source_dir,
            "{}_{}.text.txt".format(split, new5_tag)
        )
        data_to_token_ids(data, rev_class_labels, ids_path, text_path, new_vocab_path, source_dir)

