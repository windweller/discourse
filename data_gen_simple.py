"""
Investigate PTB/Wikitext2 on sentence pairs and build dataset
"""

import re
import io
import sys
import nltk
import pickle
import string
from os.path import join as pjoin


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
        sent_list = nltk.sent_tokenize(tokens)
        print("sent num: " + str(len(sent_list)))
        # prev_sent = None
        # save_to_pickle(pjoin("data", "wikitext-103", "wiki103_sent.pkl"))
        for sent in sent_list:
            words = sent.split()  # strip puncts and then split (already tokenized)
            if "but" in words[1:]:  # # no sentence from beginning has but
                but_sents.append(sent)
            if "because" in words[1:]:  # no sentence has because at beginning
                because_sents.append(sent)
            if "when" in words[1:]:  # exclude "When xxxx, xxx"
                when_sents.append(sent)
            if "if" in words[1:]:
                if_sents.append(sent)  # exclude "If xxx, xxx"
            if "for example" in sent:  # "for example ..."
                for_example_sents.append(sent)

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
    # directly use wikitext-103

    discourse_markers = ["but", "because", "when", "if", "for example"]

    wikitext_103_train_path = pjoin("data", "wikitext-103", "wiki.train.tokens")
    wikitext_103_valid_path = pjoin("data", "wikitext-103", "wiki.valid.tokens")
    wikitext_103_test_path = pjoin("data", "wikitext-103", "wiki.test.tokens")

    wikitext_103_train = list_to_dict(discourse_markers, get_wiki_pairs(wikitext_103_train_path))
    wikitext_103_valid = list_to_dict(discourse_markers, get_wiki_pairs(wikitext_103_valid_path))
    wikitext_103_test = list_to_dict(discourse_markers, get_wiki_pairs(wikitext_103_test_path))

    all_sentences_pairs = merge_dict(merge_dict(wikitext_103_train, wikitext_103_valid), wikitext_103_test)

    save_to_pickle(all_sentences_pairs, pjoin("data", "wikitext-103", "all_sentence_pairs.pkl"))

    # extension to work on Book Corpus
