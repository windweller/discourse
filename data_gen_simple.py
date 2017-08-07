"""
Investigate PTB/Wikitext2 on sentence pairs and build dataset
"""

import re
import io
import os
import sys
import nltk
import pickle
import string
import argparse
from os.path import join as pjoin

import sys
reload(sys)
sys.setdefaultencoding('utf8')

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bookcorpus", action='store_true')
    return parser.parse_args()


def write_to_file_wiki(list_sent, file_path):
    with io.open(file_path, mode="w", encoding="utf-8") as f:
        for sent in list_sent:
            f.write(sent + "\n")  # .encode("utf-8")


# def simplify(text):
#     """
#     Lowercase text and remove punctuation
#     """
#     lower = text.lower()
#     lower = lower.translate(None, string.punctuation)
#     return lower

def save_to_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def get_wiki_pairs(file_path):
    but_sents = []
    because_sents = []
    when_sents = []
    if_sents = []
    for_example_sents = []

    with io.open(file_path, 'rU', encoding="utf-8") as f:
        tokens = f.read().replace("\n", ". ")
        print("tokenizing")
        sent_list = nltk.sent_tokenize(tokens)
        print("sent num: " + str(len(sent_list)))
        # prev_sent = None
        # save_to_pickle(pjoin("data", "wikitext-103", "wiki103_sent.pkl"))
        i = 0
        for sent in sent_list:
            i += 1
            if i % 10000 == 0:
                print("reading sentence {}".format(i))
            words = sent.replace("for example", "for_example").split()  # strip puncts and then split (already tokenized)
            if "but" in words[1:]:  # # no sentence from beginning has but
                idx = words.index("but")
                but_sents.append((words[:idx], words[idx+1:]))
            if "because" in words[1:]:  # no sentence has because at beginning
                idx = words.index("because")
                because_sents.append((words[:idx], words[idx+1:]))
            if "when" in words[1:]:  # exclude "When xxxx, xxx"
                idx = words.index("when")
                when_sents.append((words[:idx], words[idx+1:]))
            if "if" in words[1:]:
                idx = words.index("if")
                if_sents.append((words[:idx], words[idx+1:]))  # exclude "If xxx, xxx"
            if "for example" in sent:  # "for example ..."
                idx = words.index("for_example")
                for_example_sents.append((words[:idx], words[idx+1:]))

    return but_sents, because_sents, when_sents, if_sents, for_example_sents


def list_to_dict(key_words, list_of_sent, print_stats=True):
    result = {}
    for i, key in enumerate(key_words):
        result[key] = list_of_sent[i]
        if print_stats:
            print("{} relation has {} sentences".format(key, len(list_of_sent[i])))
    return result


def merge_dict(dict_list1, dict_list2):
    for key, list_sent in dict_list1.iteritems():
        dict_list1[key].extend(dict_list2[key])
    return dict_list1


if __name__ == '__main__':
    args = setup_args()

    # directly use wikitext-103

    discourse_markers = ["but", "because", "when", "if", "for example"]

    if not args.bookcorpus:

        wikitext_103_train_path = pjoin("data", "wikitext-103", "wiki.train.tokens")
        wikitext_103_valid_path = pjoin("data", "wikitext-103", "wiki.valid.tokens")
        wikitext_103_test_path = pjoin("data", "wikitext-103", "wiki.test.tokens")

        print("extracting sentence pairs from train")
        wikitext_103_train = list_to_dict(discourse_markers, get_wiki_pairs(wikitext_103_train_path))
        print("extracting sentence pairs from valid")
        wikitext_103_valid = list_to_dict(discourse_markers, get_wiki_pairs(wikitext_103_valid_path))
        print("extracting sentence pairs from test")
        wikitext_103_test = list_to_dict(discourse_markers, get_wiki_pairs(wikitext_103_test_path))

        # all_sentences_pairs = merge_dict(merge_dict(wikitext_103_train, wikitext_103_valid), wikitext_103_test)

        # save_to_pickle(all_sentences_pairs, pjoin("data", "wikitext-103", "all_sentence_pairs.pkl"))
        save_to_pickle(wikitext_103_train, pjoin("data", "wikitext-103", "train.pkl"))
        save_to_pickle(wikitext_103_valid, pjoin("data", "wikitext-103", "valid.pkl"))
        save_to_pickle(wikitext_103_test, pjoin("data", "wikitext-103", "test.pkl"))

    # extension to work on Book Corpus
    else:

        bookcorpus_path = pjoin("data", "bookcorpus", "books_in_sentences")
