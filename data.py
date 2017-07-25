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
[(s1, s2, label)]
```
where `label` is class index from {0, ..., number_of_discourse_markers},
and `s1` and `s2` are lists of word ids

Also creates vocabulary file `vocab.dat` specifying the mapping between glove embeddings and ids.
"""


def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    vocab_dir = os.path.join("data", "ptb")
    glove_dir = os.path.join("data", "glove.6B")
    source_dir = os.path.join("data", "ptb")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=100, type=int)
    parser.add_argument("--random_init", action='store_true')
    parser.add_argument("--train_size", default=0.9)
    parser.add_argument("--discourse_markers", default="because,although,but,for example,when,before,after,however,so,still,though,meanwhile,while,if")
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
def process_glove(args, vocab_list, save_path, size=4e5, random_init=True):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if os.path.basename(os.path.dirname(save_path))=="winograd":
        # for winograd, steal npz from wikitext
        wikipath = pjoin(
            os.path.dirname(os.path.dirname(save_path)),
            "wikitext-103",
            os.path.basename(save_path))
        print(wikipath)
        # fix me!
        print("TO DO: RERUN FOR WIKITEXT FIRST!!")
        # stop
    if gfile.Exists(save_path + ".npz"):
        print("Glove file already exists at %s" % (save_path + ".npz"))
    else:
        glove_path = os.path.join(args.glove_dir, "glove.6B.{}d.txt".format(args.glove_dim))
        if random_init:
            glove = np.random.randn(len(vocab_list), args.glove_dim)
        else:
            glove = np.zeros((len(vocab_list), args.glove_dim))
        found = 0
        line_num = 0
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
                line_num += 1
                if line_num == size:
                    break

        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))

def create_vocabulary(vocabulary_path, sentence_pairs_data, tokenizer=None, discourse_markers=None):
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


def data_to_token_ids(data, target_path, vocabulary_path, tokenizer=None):
    if not gfile.Exists(target_path):
        vocab, _ = initialize_vocabulary(vocabulary_path)

        ids_data = []
        counter = 0
        for s1, s2, label in data:
            counter += 1
            if counter % 5000 == 0:
                print("tokenizing example %d" % counter)
            token_ids_s1 = sentence_to_token_ids(s1, vocab, tokenizer)
            token_ids_s2 = sentence_to_token_ids(s2, vocab, tokenizer)
            ids_data.append((token_ids_s1, token_ids_s2, label))

        pickle.dump(ids_data, gfile.GFile(target_path, mode="wb"))


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

    discourse_markers = args.discourse_markers.split(",")

    vocab_path = pjoin(args.vocab_dir, "vocab.dat")

    data_path = pjoin(args.source_dir, "all_sentence_pairs.pkl")
    if os.path.isfile(data_path):
        print("Loading data %s" % (str(data_path)))
        sentence_pairs_data = pickle.load(open(data_path, mode="rb"))

        create_vocabulary(vocab_path, sentence_pairs_data, tokenizer=None) # nltk.word_tokenize

        vocab, rev_vocab = initialize_vocabulary(pjoin(args.vocab_dir, "vocab.dat"))

        # ======== Trim Distributed Word Representation =======
        # If you use other word representations, you should change the code below

        process_glove(args, rev_vocab, pjoin(args.source_dir, "glove.trimmed.{}".format(args.glove_dim)),
                      random_init=args.random_init)

        # ======== Split =========

        assert(args.train_size < 1)
        split_proportions = {
            "train": args.train_size,
            "valid": (1-args.train_size)/2,
            "test": (1-args.train_size)/2
        }
        assert(sum([split_proportions[split] for split in split_proportions])==1)

        splits = {split: [] for split in split_proportions}

        # gather class labels for reference
        class_labels = {}
        i = -1

        # make split, s.t. we have similar distributions over discourse markers for each split
        for marker in discourse_markers:
            i += 1
            class_labels[marker] = i

            # shuffle sentence pairs within each marker
            # (otherwise sentences from the same document will end up in the same split)
            np.random.shuffle(sentence_pairs_data[marker])

            # add class label to each example
            all_examples = [(p[0], p[1], i) for p in sentence_pairs_data[marker]]
            total_n_examples = len(all_examples)

            # make valid and test sets (they will be equal size)
            valid_size = int(np.floor(split_proportions["valid"]*total_n_examples))
            test_size = valid_size
            splits["valid"] += all_examples[0:valid_size]
            splits["test"] += all_examples[valid_size:valid_size+test_size]

            # make train set with remaining examples
            splits["train"] += all_examples[valid_size+test_size:]

        # shuffle training set so class labels are randomized
        for split in splits: np.random.shuffle(splits[split])

        # print class labels for reference  
        print(class_labels)
        

        # ======== Creating Dataset =========

        for split in splits:
            data = splits[split]
            ids_path = pjoin(args.source_dir, split + ".ids.pkl")
            print("Tokenizing data in {}".format(split))
            data_to_token_ids(data, ids_path, vocab_path)

    else:
        print("Data file {} does not exist.".format(data_path))

