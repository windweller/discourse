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
"""


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
    parser.add_argument("--max_seq_len", default=50)
    parser.add_argument("--min_seq_len", default=5)
    parser.add_argument("--max_ratio", default=5.0)
    parser.add_argument("--undersamp_cutoff", default=50000)
    parser.add_argument("--exclude", default="")
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

        with gfile.GFile(text_path, mode="wb") as f:
            for t in shuffled_text_data:
                f.write(str([" ".join(t[0]), " ".join(t[1]), rev_class_labels[t[2]]]) + "\n")


def filter_examples(orig_pairs, class_label, max_seq_len, min_seq_len, max_ratio, undersamp_cutoff):
    new_pairs_with_labels = []
    min_ratio = 1 / max_ratio
    for s1, s2 in orig_pairs:
        ratio = float(len(s1))/len(s2)
        if len(s1) < min_seq_len or max_seq_len < len(s1):
            pass
        elif len(s2) < min_seq_len or max_seq_len < len(s2):
            pass
        elif ratio < min_ratio or max_ratio < ratio:
            pass
        else:
            new_pairs_with_labels.append((s1, s2, class_label))

    # shuffle sentence pairs within each marker
    # (otherwise sentences from the same document will end up in the same split)
    np.random.shuffle(new_pairs_with_labels)
    return new_pairs_with_labels[:undersamp_cutoff]

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

    discourse_markers = [d for d in [
        "because", "although",
        "but", "when",             # "for_example"
        "before", "after", "however", "so", "still", "though",
        "meanwhile",
        "while", "if"
    ] if d not in args.exclude.split(",")]

    vocab_path = pjoin(args.vocab_dir, "vocab.dat")

    data_path = pjoin(args.source_dir, "all_sentence_pairs.pkl")

    if os.path.isfile(data_path):
        print("Loading data %s" % (str(data_path)))
        sentence_pairs_data = pickle.load(open(data_path, mode="rb"))

        create_vocabulary(vocab_path, sentence_pairs_data)

        vocab, rev_vocab = initialize_vocabulary(pjoin(args.vocab_dir, "vocab.dat"))

        # ======== Trim Distributed Word Representation =======
        # If you use other word representations, you should change the code below

        process_glove(args, vocab, pjoin(args.source_dir, "glove.trimmed.{}".format(args.glove_dim)),
                      random_init=args.random_init)

        # ======== Split =========

        assert(args.train_size < 1 and args.train_size > 0)
        split_proportions = {
            "train": args.train_size,
            "valid": (1-args.train_size)/2,
            "test": (1-args.train_size)/2
        }
        assert(sum([split_proportions[split] for split in split_proportions])==1)

        splits = {split: {} for split in split_proportions}

        # gather class labels for reference
        class_labels = {}
        rev_class_labels = []
        i = -1

        # make split, s.t. we have similar distributions over discourse markers for each split
        for marker in discourse_markers:
            i += 1
            class_labels[marker] = i
            rev_class_labels.append(marker)

            # add class label to each example
            all_examples = filter_examples(
                sentence_pairs_data[marker],
                i,
                args.max_seq_len,
                args.min_seq_len,
                args.max_ratio,
                args.undersamp_cutoff
            )
            total_n_examples = len(all_examples)
            print("total number of {}: {}".format(marker, total_n_examples))

            # make valid and test sets (they will be equal size)
            valid_size = int(np.floor(split_proportions["valid"]*total_n_examples))
            test_size = valid_size
            splits["valid"][marker] = all_examples[0:valid_size]
            splits["test"][marker] = all_examples[valid_size:valid_size+test_size]

            # make train set with remaining examples
            splits["train"][marker] = all_examples[valid_size+test_size:]

        # print class labels for reference  
        pickle.dump(class_labels, open(pjoin(args.source_dir, "class_labels.pkl"), "wb"))
        

        # ======== Creating Dataset =========

        for split in splits:
            data = splits[split]
            print("Converting data in {}".format(split))
            if args.exclude == "":
                tag = "all"
            else:
                tag = "no_" + args.exclude.replace(",", "_")
            ids_path = pjoin(
                args.source_dir,
                "{}_{}.ids.pkl".format(split, tag)
            )
            text_path = pjoin(
                args.source_dir,
                "{}_{}.text.txt".format(split, tag)
            )

            data_to_token_ids(data, rev_class_labels, ids_path, text_path, vocab_path, args.source_dir)

    else:
        print("Data file {} does not exist.".format(data_path))

