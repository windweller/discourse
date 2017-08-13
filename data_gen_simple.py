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
    parser.add_argument("--sentence_initial", action='store_true')
    parser.add_argument("--split", default='orig')
    parser.add_argument("--markers", default='five')
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

def undo_rephrase(lst):
    return " ".join(lst).replace("for_example", "for example").split()

def rephrase(str):
    return str.replace("for example", "for_example")

def get_books_pairs(file_path, discourse_markers, sentence_initial=False):
    sents = {d: [] for d in discourse_markers}

    total_pairs_extracted = 0
    with io.open(file_path, 'rU', encoding="utf-8") as f:
        prev_words = None
        i = 0
        for line in f:
            sent = line[:-1]
            i += 1
            if i % 100000 == 0:
                print("reading sentence {}".format(i))
            if total_pairs_extracted >= 200:
                break
            words = rephrase(sent).split()  # strip puncts and then split (already tokenized)
            # all of these have try statements, because sometimes the discourse marker will
            # only be a part of the word, and so it won't show up in the words list
            for marker in discourse_markers:
                if marker == "for example":
                    proxy_marker = "for_example"
                else:
                    proxy_marker = marker
                if proxy_marker in words[1:]: # sentence-internal
                    idx = words.index(proxy_marker)
                    sents[marker].append((undo_rephrase(words[:idx]), undo_rephrase(words[idx+1:])))
                    total_pairs_extracted += 1
                elif sentence_initial and marker in ["but", "because"] and prev_words!=None and words[0].lower()==marker:
                    sents[marker].append(prev_words, undo_rephrase(words[1:]))
                    total_pairs_extracted += 1
                elif sentence_initial and proxy_marker in ["for_example"] and prev_words!=None and sent[:11].lower()=="for example":
                    sents[marker].append(prev_words, undo_rephrase(words[2:]))
                    total_pairs_extracted += 1

            prev_words = sent.split()

    return sents

def get_wiki_pairs(file_path, discourse_markers, sentence_initial=False):
    sents = {d: [] for d in discourse_markers}

    with io.open(file_path, 'rU', encoding="utf-8") as f:
        tokens = f.read().replace("\n", ". ")
        print("tokenizing")
        sent_list = nltk.sent_tokenize(tokens)
        print("sent num: " + str(len(sent_list)))
        prev_words = None
        # save_to_pickle(pjoin("data", "wikitext-103", "wiki103_sent.pkl"))
        i = 0
        for sent in sent_list:
            i += 1
            if i % 100000 == 0:
                print("reading sentence {}".format(i))
            words = rephrase(sent).split()  # strip puncts and then split (already tokenized)
            # all of these have try statements, because sometimes the discourse marker will
            # only be a part of the word, and so it won't show up in the words list
            for marker in discourse_markers:
                if marker == "for example":
                    proxy_marker = "for_example" 
                else:
                    proxy_marker = marker
                if proxy_marker in words[1:]: # sentence-internal
                    idx = words.index(proxy_marker)
                    sents[marker].append((undo_rephrase(words[:idx]), undo_rephrase(words[idx+1:])))
                elif sentence_initial and marker in ["but", "because"] and prev_words!=None and words[0].lower()==marker:
                    sents[marker].append(prev_words, undo_rephrase(words[1:]))
                elif sentence_initial and proxy_marker in ["for_example"] and prev_words!=None and sent[:11].lower()=="for example":
                    sents[marker].append(prev_words, undo_rephrase(words[2:]))

            prev_words = sent.split()

    return sents


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

    all_discourse_markers = [
        "because", "although",
        "but", "when", "for example",
        "before", "after", "however", "so", "still", "though",
        "meanwhile",
        "while", "if"
    ]
    five_discourse_markers = ["but", "because", "when", "if", "for example"]
    if args.markers == "all":
        discourse_markers = all_discourse_markers
    elif args.markers == "five":
        discourse_markers = five_discourse_markers
    else:
        raise ("error in discourse markerException set")


    if not args.bookcorpus:

        wikitext_103_train_path = pjoin("data", "wikitext-103", "wiki.train.tokens")
        wikitext_103_valid_path = pjoin("data", "wikitext-103", "wiki.valid.tokens")
        wikitext_103_test_path = pjoin("data", "wikitext-103", "wiki.test.tokens")

        print("extracting sentence pairs from train")
        wikitext_103_train = get_wiki_pairs(wikitext_103_train_path, discourse_markers, sentence_initial=args.sentence_initial)
        print("extracting sentence pairs from valid")
        wikitext_103_valid = get_wiki_pairs(wikitext_103_valid_path, discourse_markers, sentence_initial=args.sentence_initial)
        print("extracting sentence pairs from test")
        wikitext_103_test = get_wiki_pairs(wikitext_103_test_path, discourse_markers, sentence_initial=args.sentence_initial)


        if args.split == "orig":
            save_to_pickle(wikitext_103_train, pjoin("data", "wikitext-103", "train.pkl"))
            save_to_pickle(wikitext_103_valid, pjoin("data", "wikitext-103", "valid.pkl"))
            save_to_pickle(wikitext_103_test, pjoin("data", "wikitext-103", "test.pkl"))
        elif split == "rand":
            all_sentences_pairs = merge_dict(merge_dict(wikitext_103_train, wikitext_103_valid), wikitext_103_test)
            save_to_pickle(all_sentences_pairs, pjoin("data", "wikitext-103", "all_sentence_pairs.pkl"))
        else:
            raise Exception("error in train/valid/test split option")

    # extension to work on Book Corpus
    else:

        bookcorpus_path = pjoin("data", "books", "books_large_p1.txt")
        all_sentences_pairs = get_books_pairs(bookcorpus_path, discourse_markers, sentence_initial=args.sentence_initial)
        save_to_pickle(all_sentences_pairs, pjoin("data", "books", "all_sentence_pairs_string_ssplit.pkl"))
