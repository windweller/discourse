#! /usr/bin/env python

import numpy as np
import argparse
import io
import nltk
import pickle

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

# patterns = {
#     "because": ("IN", "mark", "advcl"),
# }

def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument("--dataset", default="wikitext-103", type=str)
    parser.add_argument("--data_tag", default="", type=str)
    parser.add_argument("--train_size", default=0.9, type=float)
    parser.add_argument("--method", default="string_ssplit_int_init", type=str)
    parser.add_argument("--caching", action='store_true')
    parser.add_argument("--action", default='collect_raw', type=str)
    return parser.parse_args()

def undo_rephrase(lst):
    return " ".join(lst).replace("for_example", "for example").split()

def rephrase(str):
    return str.replace("for example", "for_example")

def string_ssplit_int_init(source_dir, split, train_size):

    def get_data(source_dir, marker, sentence_type, split, train_size):
        filename = "{}_raw_{}_{}_{}.txt".format(marker, sentence_type, split, train_size)
        file_path = pjoin(source_dir, "raw", "split", filename)
        return open(file_path, "rU").readlines()

    data = {"s1": [], "s2": [], "label": []}
    for marker in DISCOURSE_MARKERS:
        sentences = get_data(source_dir, marker, "s", split, train_size)
        previous = get_data(source_dir, marker, "prev", split, train_size)
        assert(len(sentences) == len(previous))

        for i in range(len(sentences)):
            sentence = sentences[i]
            previous_sentence = previous[i]

            if marker=="for example":
                words = rephrase(sentence).split()
                if "for_example"==words[0].lower():
                    s1 = previous_sentence
                    s2 = " ".join(undo_rephrase(words[1:]))
                else:
                    idx = [w.lower() for w in words].index("for_example")
                    s1 = " ".join(undo_rephrase(words[:idx]))
                    s2 = " ".join(undo_rephrase(words[idx+1:]))
            else:
                words = sentence.split()
                if marker==words[0].lower(): # sentence-initial
                    s1 = previous_sentence
                    s2 = " ".join(words[1:])
                else: # sentence-internal
                    idx = [w.lower() for w in words].index(marker)
                    s1 = " ".join(words[:idx])
                    s2 = " ".join(words[idx+1:])
            data["label"].append(marker)
            data["s1"].append(s1.strip())
            data["s2"].append(s2.strip())

    return data

def string_ssplit_clean_markers():
    raise Exception("haven't included clean ssplit in this script yet")

def depparse_ssplit_v1():
    raise Exception("haven't included old combination depparse ssplit in this script yet")

def depparse_ssplit_v2():
    raise Exception("haven't included new depparse ssplit in this script yet")

def collect_raw_sentences(source_dir, dataset, caching):
    raw_dir = pjoin(source_dir, "raw")
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    collection_dir = pjoin(raw_dir, "collection")
    if not os.path.exists(collection_dir):
        os.makedirs(collection_dir)

    if dataset == "wikitext-103":
        filenames = [
            "wiki.train.tokens",
            "wiki.valid.tokens", 
            "wiki.test.tokens"
        ]
    else:
        raise Exception("not implemented")

    sentences = {marker: {"sentence": [], "previous": []} for marker in DISCOURSE_MARKERS}
    
    for filename in filenames:
        print("reading {}".format(filename))
        file_path = pjoin(source_dir, filename)
        with io.open(file_path, 'rU', encoding="utf-8") as f:
            # tokenize sentences
            sentences_cache_file = file_path + ".CACHE_SENTS"
            if caching and os.path.isfile(sentences_cache_file):
                sent_list = pickle.load(open(sentences_cache_file, "rb"))
            else:
                tokens = f.read().replace("\n", ". ")
                print("tokenizing")
                sent_list = nltk.sent_tokenize(tokens)
                if caching:
                    pickle.dump(sent_list, open(sentences_cache_file, "wb"))

        # check each sentence for discourse markers
        previous_sentence = ""
        for sentence in sent_list:
            words = rephrase(sentence).split()  # replace "for example"
            for marker in DISCOURSE_MARKERS:
                if marker == "for example":
                    proxy_marker = "for_example" 
                else:
                    proxy_marker = marker

                if proxy_marker in words:
                    sentences[marker]["sentence"].append(sentence)
                    sentences[marker]["previous"].append(previous_sentence)
            previous_sentence = sentence

    print('writing files')
    for marker in sentences:
        sentence_path = pjoin(collection_dir, "{}_raw_s.txt".format(marker))
        previous_path = pjoin(collection_dir, "{}_raw_prev.txt".format(marker))
        with open(sentence_path, "w") as sentence_file:
            for s in sentences[marker]["sentence"]:
                sentence_file.write(s + "\n")
        with open(previous_path, "w") as previous_file:
            for s in sentences[marker]["previous"]:
                previous_file.write(s + "\n")

def split_raw(source_dir, train_size):
    assert(train_size < 1 and train_size > 0)

    split_dir = pjoin(source_dir, "raw", "split")
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    for marker in DISCOURSE_MARKERS:
        sentences = open(pjoin(source_dir, "raw", "collection", "{}_raw_s.txt".format(marker)), "rU").readlines()
        previous_sentences = open(pjoin(source_dir, "raw", "collection", "{}_raw_prev.txt".format(marker)), "rU").readlines()
        assert(len(sentences)==len(previous_sentences))

        indices = range(len(sentences))
        np.random.shuffle(indices)

        n_test = len(indices) * train_size
        n_valid = n_test
        n_train = len(indices) - n_test - n_valid

        splits = {split: {"s": [], "prev": []} for split in ["train", "valid", "test"]}

        for i in range(len(indices)):
            sentence_index = indices[i]
            sentence = sentences[sentence_index]
            previous = previous_sentences[sentence_index]
            if i<n_test:
                split="test"
            elif i<(n_test + n_valid):
                split="valid"
            else:
                split="train"
            splits[split]["s"].append(sentence)
            splits[split]["prev"].append(previous)

        for split in splits:
            for sentence_type in ["s", "prev"]:
                write_path = pjoin(split_dir, "{}_raw_{}_{}_{}.txt".format(marker, sentence_type, split, train_size))
                with open(write_path, "w") as write_file:
                    for sentence in splits[split][sentence_type]:
                        write_file.write(sentence)

def ssplit(method, source_dir, data_tag, train_size):
    methods = {
        "string_ssplit_int_init": string_ssplit_int_init,
        "string_ssplit_clean_markers": string_ssplit_clean_markers,
        "depparse_ssplit_v1": depparse_ssplit_v1
    }
    assert(args.method in methods)

    # (a dictionary {train: {...}, valid: {...}, test: {...}})
    splits = {split: methods[args.method](source_dir, split, train_size) for split in ["train", "valid", "test"]}

    if data_tag == "":
        extra_tag = ""
    else:
        extra_tag = "_" + data_tag

    sub_directory = "{}_train{}{}".format(
        args.method,
        train_size,
        extra_tag
    )

    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)

    for split in splits:
        # randomize the order at this point
        labels = splits[split]["label"]
        s1 = splits[split]["s1"]
        s2 = splits[split]["s2"]

        assert(len(labels) == len(s1) and len(s1) == len(s2))
        indices = range(len(labels))
        np.random.shuffle(indices)

        for element_type in ["label", "s1", "s2"]:
            filename = sub_directory + "_{}_{}.txt".format(split, element_type)
            file_path = pjoin(source_dir, sub_directory, filename)
            with open(file_path, "w") as write_file:
                for index in indices:
                    element = splits[split][element_type][index]
                    write_file.write(element + "\n")


if __name__ == '__main__':
    args = setup_args()

    source_dir = os.path.join("data", args.dataset)

    if args.action == "collect_raw":
        collect_raw_sentences(source_dir, args.dataset, args.caching)
    elif args.action == "split":
        split_raw(source_dir, args.train_size)
    elif args.action == "ssplit":
        ssplit(args.method, source_dir, args.data_tag, args.train_size)


