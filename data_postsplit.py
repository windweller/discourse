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


def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    glove_dir = os.path.join("data", "glove.6B")
    parser.add_argument("--prefix", default="", type=str)
    parser.add_argument("--dataset", default="wikitext-103", type=str)
    parser.add_argument("--data_name", default="all_sentence_pairs.pkl", type=str)
    parser.add_argument("--data_tag", default="", type=str)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_dim", default=300, type=int)
    parser.add_argument("--random_init", action='store_true')
    parser.add_argument("--max_seq_len", default=50, type=int)
    parser.add_argument("--min_seq_len", default=5, type=int)
    parser.add_argument("--max_ratio", default=5.0, type=float)
    parser.add_argument("--undersamp_cutoff", default=50000, type=int)
    parser.add_argument("--no_cutoff", action='store_true')
    parser.add_argument("--exclude", default="")
    parser.add_argument("--include", default="")
    parser.add_argument("--train_size", default=0.9, type=float)
    parser.add_argument("--method", default="string_ssplit_int_init", type=str)

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


def data_to_token_ids(data, class_labels, rev_class_labels, target_path, text_path, vocabulary_path, data_dir):
    if gfile.Exists(target_path):
        print("file {} already exists".format(target_path))
    else:
        vocab, _ = initialize_vocabulary(vocabulary_path)

        ids_data = []
        for marker in data:
            counter = 0
            for s1, s2 in data[marker]:
                label = class_labels[marker]
                counter += 1
                if counter % 100000 == 0:
                    print("converting %s %d" % (marker, counter))
                token_ids_s1 = sentence_to_token_ids(s1, vocab)
                token_ids_s2 = sentence_to_token_ids(s2, vocab)
                ids_data.append((token_ids_s1, token_ids_s2, label))

        # shuffle so markers aren't all clumped together
        shuffled_idx = range(len(ids_data))
        np.random.shuffle(shuffled_idx)

        shuffled_ids_data = [ids_data[idx] for idx in shuffled_idx]
        shuffled_text_data = [text_data[idx] for idx in shuffled_idx]

        print("writing {} and {}".format(target_path, text_path))
        pickle.dump(shuffled_ids_data, gfile.GFile(target_path, mode="wb"))


def filter_examples(orig_pairs, class_label, max_seq_len, min_seq_len, max_ratio, undersamp_cutoff):
    new_pairs_with_labels = []
    min_ratio = 1 / max_ratio
    for s1, s2 in orig_pairs:
        ratio = float(len(s1))/max(len(s2), 0.0001)
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
    if args.no_cutoff:
        return new_pairs_with_labels
    return new_pairs_with_labels[:undersamp_cutoff]

def tokenize_sentence_pair_data(sentence_pairs_data):
    tokenized_sent_pair_data = {}
    for key, value in sentence_pairs_data.iteritems():
        sent_pairs = []
        for sent_pair in value:
            sent_pairs.append((sent_pair[0].split(), sent_pair[1].split()))
        tokenized_sent_pair_data[key] = sent_pairs

    return tokenized_sent_pair_data


def merge_dict(dict_list1, dict_list2):
    for key, list_sent in dict_list1.iteritems():
        dict_list1[key].extend(dict_list2[key])
    return dict_list1

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

    vocab_dir = os.path.join(args.prefix, "data", args.dataset)
    source_dir = os.path.join(args.prefix, "data", args.dataset)

    # ======== Get discourse marker set from args =======

    assert(args.include=="" or args.exclude=="")

    all_discourse_markers = [
        "because", "although",
        "but", "when", "for example",
        "before", "after", "however", "so", "still", "though",
        "meanwhile",
        "while", "if"
    ]
    if args.include=="":
        # --exclude "because,for example,algorithm"
        discourse_markers = [d for d in all_discourse_markers if d not in args.exclude.split(",")]
    else:
        discourse_markers = args.include.split(",")
        assert(all([d in all_discourse_markers for d in discourse_markers]))

    if args.exclude == "" and args.include == "":
        tag = "all"
    elif args.exclude != "":
        tag = "no_" + args.exclude.replace(",", "_").replace(" ", "_")
        # last part is for "for example"
    elif args.include != "":
        tag = args.include.replace(",", "_").replace(" ", "_")
    else:
        raise Exception("no match state for exclude/include")

    # ======== Load data =======
    print("Loading data")

    if args.data_tag == "":
        extra_tag = ""
    else:
        extra_tag = "_" + args.data_tag

    splits = {}
    for split in ["train", "valid", "test"]:
        filename = pjoin(source_dir, "{}_allpairs_allmarkers_{}_train{}{}.pkl".format(
            split,
            args.method,
            args.train_size,
            extra_tag
        ))
        data = pickle.load(open(filename, "rb"))
        splits[split] = data

    # ======== Setup vocabulary =======

    combined_data = merge_dict(splits["train"], merge_dict(splits["valid"], splits["test"]))
    vocab_path = pjoin(vocab_dir, "vocab_{}.dat".format(tag))
    create_vocabulary(vocab_path, combined_data)
    vocab, rev_vocab = initialize_vocabulary(pjoin(vocab_dir, "vocab_{}.dat".format(tag)))

    # ======== Trim Distributed Word Representation =======
    # If you use other word representations, you should change the code below

    process_glove(args, vocab, pjoin(source_dir, "glove.trimmed.{}_{}.npz".format(args.glove_dim, tag)),
                  random_init=args.random_init)

    # ======== Create class labels =======

    # gather class labels
    class_labels = {}
    rev_class_labels = []
    i = -1
    for marker in discourse_markers:
        i += 1
        class_labels[marker] = i
        rev_class_labels.append(marker)
    # print class labels for reference  
    pickle.dump(class_labels, open(pjoin(source_dir, "class_labels_{}.pkl".format(tag)), "wb"))
    pickle.dump(rev_class_labels, open(pjoin(source_dir, "rev_class_labels_{}.pkl".format(tag)), "wb"))

    # ======== Creating Dataset =========

    for split in splits:
        data = splits[split]
        print("Converting data in {}".format(split))
        ids_path = pjoin(
            source_dir,
            "{}_{}.ids.pkl".format(split, tag)
        )
        data_to_token_ids(data, class_labels, rev_class_labels, ids_path, vocab_path, source_dir)


