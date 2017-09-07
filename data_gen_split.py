#! /usr/bin/env python

import numpy as np
import argparse
import io
import nltk

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
from os.path import join as pjoin

np.random.seed(123)

DISCOURSE_MARKERS = [
    "after",
    "also",
    "although",
    "and",
    "as",
    "because",
    "before",
    "but",
    "for example",
    "however",
    "if",
    "meanwhile",
    "so",
    "still",
    "then",
    "though",
    "when",
    "while"
]

def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument("--dataset", default="wikitext-103", type=str)
    parser.add_argument("--data_tag", default="", type=str)
    parser.add_argument("--train_size", default=0.9, type=float)
    parser.add_argument("--method", default="string_ssplit_int_init", type=str)
    parser.add_argument("--caching", action='store_true')
    return parser.parse_args()

def undo_rephrase(lst):
    return " ".join(lst).replace("for_example", "for example").split()

def rephrase(str):
    return str.replace("for example", "for_example")

def get_wiki_pairs(file_path, sentence_initial=False, caching=False):
    sents = {d: [] for d in DISCOURSE_MARKERS}

    with io.open(file_path, 'rU', encoding="utf-8") as f:

        # tokenize sentences
        sentences_cache_file = pjoin("data", "wikitext-103", "wiki103_sent.pkl")
        if caching and os.path.isfile(sentences_cache_file):
            sent_list = pickle.load(open(sentences_cache_file, "rb"))
        else:
            tokens = f.read().replace("\n", ". ")
            print("tokenizing")
            sent_list = nltk.sent_tokenize(tokens)
            print("sent num: " + str(len(sent_list)))
            if caching:
                save_to_pickle(sentences_cache_file)

        # check each sentence for discourse markers and collect sentence pairs
        prev_words = None
        i = 0
        for sent in sent_list:
            i += 1
            if i % 100000 == 0:
                print("reading sentence {}".format(i))
            words = rephrase(sent).split()  # strip puncts and then split (already tokenized)
            # all of these have try statements, because sometimes the discourse marker will
            # only be a part of the word, and so it won't show up in the words list
            for marker in DISCOURSE_MARKERS:
                if marker == "for example":
                    proxy_marker = "for_example" 
                else:
                    proxy_marker = marker
                if proxy_marker in words[1:]: # sentence-internal
                    idx = words.index(proxy_marker)
                    sents[marker].append((undo_rephrase(words[:idx]), undo_rephrase(words[idx+1:])))
                elif sentence_initial and marker in ["but", "because"] and prev_words!=None and words[0].lower()==marker:
                    sents[marker].append((prev_words, undo_rephrase(words[1:])))
                elif sentence_initial and proxy_marker in ["for_example"] and prev_words!=None and sent[:11].lower()=="for example":
                    sents[marker].append((prev_words, undo_rephrase(words[2:])))

            prev_words = sent.split()

    return sents

def string_ssplit_int_init(dataset, caching=False):

    if dataset == "wikitext-103":

        wikitext_103_train_path = pjoin("data", "wikitext-103", "wiki.train.tokens")
        wikitext_103_valid_path = pjoin("data", "wikitext-103", "wiki.valid.tokens")
        wikitext_103_test_path = pjoin("data", "wikitext-103", "wiki.test.tokens")

        print("extracting sentence pairs from train")
        wikitext_103_train = get_wiki_pairs(wikitext_103_train_path, sentence_initial=True, caching=caching)
        print("extracting sentence pairs from valid")
        wikitext_103_valid = get_wiki_pairs(wikitext_103_valid_path, sentence_initial=True, caching=caching)
        print("extracting sentence pairs from test")
        wikitext_103_test = get_wiki_pairs(wikitext_103_test_path, sentence_initial=True, caching=caching)

    elif dataset == "books":
        raise Exception("haven't written how to parse dataset {}".format(dataset))
    else:
        raise Exception("don't know how to parse dataset {}".format(dataset))

def string_ssplit_clean_markers():
    raise Exception("haven't included clean ssplit in this script yet")

def depparse_ssplit_v1():
    raise Exception("haven't included depparse ssplit in this script yet")

def split_dictionary(data_dict, train_size):
    split_proportions = {
        "train": args.train_size,
        "valid": (1-args.train_size)/2,
        "test": (1-args.train_size)/2
    }
    assert(sum([split_proportions[split] for split in split_proportions])==1)
    splits = {split: {} for split in split_proportions}

    n_total = 0
    for marker in data_dict:
        examples_for_this_marker = data_dict[marker]
        n_marker = len(examples_for_this_marker)
        n_total += n_marker

        print("number of examples for {}: {}".format(marker, n_marker))

        # make valid and test sets (they will be equal size)
        valid_size = int(np.floor(split_proportions["valid"]*n_marker))
        test_size = valid_size
        splits["valid"][marker] = examples_for_this_marker[0:valid_size]
        splits["test"][marker] = examples_for_this_marker[valid_size:valid_size+test_size]
        # make train set with remaining examples
        splits["train"][marker] = examples_for_this_marker[valid_size+test_size:]

    print("total number of examples: {}".format(n_total))

    return splits


if __name__ == '__main__':
    args = setup_args()

    source_dir = os.path.join("data", args.dataset)

    # collect sentence pairs
    methods = {
        "string_ssplit_int_init": string_ssplit_int_init,
        "string_ssplit_clean_markers": string_ssplit_clean_markers,
        "depparse_ssplit_v1": depparse_ssplit_v1,
        "split_dictionary": split_dictionary
    }
    data_dict = methods[args.method](args.dataset, caching=args.caching)

    assert(args.train_size < 1 and args.train_size > 0)
    assert(args.method in methods)

    # split train, valid, test
    # (returns a dictionary {train: {...}, valid: {...}, test: {...}})
    splits = split_dictionary(data_dict, args.train_size)

    if args.data_tag == "":
        extra_tag = ""
    else:
        extra_tag = "_" + args.data_tag

    for split in splits:
        data = splits[split]
        filename = "{}_all_pairs_{}_train{}{}".format(
            split,
            args.method,
            args.train_size,
            extra_tag
        )
        pickle.dump(data, open(filename, "wb"))


