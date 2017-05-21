"""
Investigate PTB/Wikitext2 on sentence pairs and build dataset
"""

import re
import io
import sys
import nltk
from os.path import join as pjoin


def write_to_file(list_sent, file_path):
    with open(file_path, mode="wb") as f:
        for sent in list_sent:
            f.write(sent)  # both PTB and Wikitext has \n already!


def write_to_file_wiki(list_sent, file_path):
    with io.open(file_path, mode="w", encoding="utf-8") as f:
        for sent in list_sent:
            f.write(sent)  # .encode("utf-8")


def get_pairs(file_path):
    but_sents = []
    cause_sents = []
    with open(file_path, 'r') as f:
        for line in f:
            if "but" in line:
                words = line.split()
                if "but" in words[1:]:  # no sentence from beginning has but
                    but_sents.append(line)
            if "because" in line:
                cause_sents.append(line)

    return but_sents, cause_sents


def get_wiki_pairs(file_path):
    but_sents = []
    cause_sents = []
    with io.open(file_path, 'rU', encoding="utf-8") as f:
        tokens = f.read().replace("\n", ". ")
        sent_list = nltk.sent_tokenize(tokens)
        print("sent num: " + str(len(sent_list)))
        for sent in sent_list:
            if "but" in sent:
                words = sent.split()
                if "but" in words[1:]:  # no sentence from beginning has but
                    but_sents.append(sent)
            if "because" in sent:
                cause_sents.append(sent)

    return but_sents, cause_sents


if __name__ == '__main__':
    ptb_train_path = pjoin("data", "ptb", "ptb.train.txt")
    ptb_valid_path = pjoin("data", "ptb", "ptb.valid.txt")
    ptb_test_path = pjoin("data", "ptb", "ptb.test.txt")

    ptb_train = get_pairs(ptb_train_path)
    ptb_valid = get_pairs(ptb_valid_path)
    ptb_test = get_pairs(ptb_test_path)

    print(len(ptb_train[0]), len(ptb_train[1]))

    write_to_file(ptb_train[0], pjoin("data", "ptb", "train_BUT.txt"))
    write_to_file(ptb_train[1], pjoin("data", "ptb", "train_BECAUSE.txt"))

    write_to_file(ptb_valid[0], pjoin("data", "ptb", "valid_BUT.txt"))
    write_to_file(ptb_valid[1], pjoin("data", "ptb", "valid_BECAUSE.txt"))

    write_to_file(ptb_test[0], pjoin("data", "ptb", "test_BUT.txt"))
    write_to_file(ptb_test[1], pjoin("data", "ptb", "test_BECAUSE.txt"))

    # this is in-sentence transition

    # wikitext_2_train_path = pjoin("data", "wikitext-2", "wiki.train.tokens")
    # wikitext_2_valid_path = pjoin("data", "wikitext-2", "wiki.valid.tokens")
    # wikitext_2_test_path = pjoin("data", "wikitext-2", "wiki.test.tokens")
    #
    # wikitext_2_train = get_wiki_pairs(wikitext_2_train_path)
    # wikitext_2_valid = get_wiki_pairs(wikitext_2_valid_path)
    # wikitext_2_test = get_wiki_pairs(wikitext_2_test_path)
    #
    # print(len(wikitext_2_train[0]), len(wikitext_2_train[1]))

    # directly use wikitext-103

    wikitext_103_train_path = pjoin("data", "wikitext-103", "wiki.train.tokens")
    wikitext_103_valid_path = pjoin("data", "wikitext-103", "wiki.valid.tokens")
    wikitext_103_test_path = pjoin("data", "wikitext-103", "wiki.test.tokens")

    wikitext_103_train = get_wiki_pairs(wikitext_103_train_path)
    wikitext_103_valid = get_wiki_pairs(wikitext_103_valid_path)
    wikitext_103_test = get_wiki_pairs(wikitext_103_test_path)

    print(len(wikitext_103_train[0]), len(wikitext_103_train[1]))

    write_to_file_wiki(wikitext_103_train[0], pjoin("data", "wikitext-103", "train_BUT.txt"))
    write_to_file_wiki(wikitext_103_train[1], pjoin("data", "wikitext-103", "train_BECAUSE.txt"))

    write_to_file_wiki(wikitext_103_valid[0], pjoin("data", "wikitext-103", "valid_BUT.txt"))
    write_to_file_wiki(wikitext_103_valid[1], pjoin("data", "wikitext-103", "valid_BECAUSE.txt"))

    write_to_file_wiki(wikitext_103_test[0], pjoin("data", "wikitext-103", "test_BUT.txt"))
    write_to_file_wiki(wikitext_103_test[1], pjoin("data", "wikitext-103", "test_BECAUSE.txt"))
