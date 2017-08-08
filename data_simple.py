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
    parser.add_argument("--exclude", default="")
    parser.add_argument("--include", default="")
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


def merge_dict(dict_list1, dict_list2):
    for key, list_sent in dict_list1.iteritems():
        dict_list1[key].extend(dict_list2[key])
    return dict_list1

def data_to_token_ids(data, discourse_markers, class_labels, rev_class_labels, target_path, text_path, vocabulary_path, data_dir):
    if gfile.Exists(target_path):
        print("file {} already exists".format(target_path))
    else:
        vocab, _ = initialize_vocabulary(vocabulary_path)

        # fix me: this will be a list instead
        ids_data = []
        text_data = []
        for marker in discourse_markers:
            # ids_data[marker] = []
            counter = 0
            for s1, s2 in data[marker]:
                label = class_labels[marker]
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



def tokenize_sentence_pair_data(sentence_pairs_data):
    tokenized_sent_pair_data = {}
    for key, value in sentence_pairs_data.iteritems():
        sent_pairs = []
        for sent_pair in value:
            sent_pairs.append((sent_pair[0].split(), sent_pair[1].split()))
        tokenized_sent_pair_data[key] = sent_pairs

    return tokenized_sent_pair_data


if __name__ == '__main__':
    args = setup_args()
    assert(args.include=="" or args.exclude=="")

    all_discourse_markers = ["but", "because", "when", "if", "for example"]

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

    vocab_path = pjoin(args.vocab_dir, "vocab_{}.dat".format(tag))

    train_path = pjoin("data", "wikitext-103", "train.pkl")
    valid_path = pjoin("data", "wikitext-103", "valid.pkl")
    test_path = pjoin("data", "wikitext-103", "test.pkl")

    if all([os.path.isfile(data_path) for data_path in [train_path, valid_path, test_path]]):

        print("Loading data")
        train_data = pickle.load(open(train_path, mode="rb"))
        valid_data = pickle.load(open(valid_path, mode="rb"))
        test_data = pickle.load(open(test_path, mode="rb"))

        sentence_pairs_data = merge_dict(merge_dict(train_data, valid_data), test_data)

        create_vocabulary(vocab_path, sentence_pairs_data)

        vocab, rev_vocab = initialize_vocabulary(pjoin(args.vocab_dir, "vocab_{}.dat".format(tag)))

        # ======== Trim Distributed Word Representation =======
        # If you use other word representations, you should change the code below

        process_glove(args, vocab, pjoin(args.source_dir, "glove.trimmed.{}_{}.npz".format(args.glove_dim, tag)),
                      random_init=args.random_init)

        # ======== Creating Dataset =========

        splits = {
            "train": train_data,
            "valid": valid_data,
            "test": test_data
        }

        class_labels = {discourse_markers[i]: i for i in range(len(discourse_markers))}

        # print class labels for reference  
        pickle.dump(class_labels, open(pjoin(args.source_dir, "class_labels_{}.pkl".format(tag)), "wb"))
        # print class labels for reference  
        # pickle.dump(discourse_markers, open(pjoin(args.source_dir, "class_labels_list_{}.pkl".format(tag)), "wb"))

        for split in ["train", "valid", "test"]:
            data = splits[split]
            print("Converting data in {}".format(split))
            ids_path = pjoin(
                args.source_dir,
                "{}_{}.ids.pkl".format(split, tag)
            )
            text_path = pjoin(
                args.source_dir,
                "{}_{}.text.txt".format(split, tag)
            )

            data_to_token_ids(data, discourse_markers, class_labels, discourse_markers, ids_path, text_path, vocab_path, args.source_dir)

    else:
        if not os.path.isfile(train_path):
            print("Data file {} does not exist.".format(train_path))
        if not os.path.isfile(valid_path):
            print("Data file {} does not exist.".format(valid_path))
        if not os.path.isfile(test_path):
            print("Data file {} does not exist.".format(test_path))
