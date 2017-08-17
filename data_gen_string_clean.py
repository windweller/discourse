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
    parser.add_argument("--segment_index", default=0, type=int)
    parser.add_argument("--n_segments", default=1, type=int)
    parser.add_argument("--cutoff", default=None, type=int) #300000
    parser.add_argument("--subsample", action='store_true')
    parser.add_argument("--aggregate", action='store_true')
    return parser.parse_args()


def write_to_file_wiki(list_sent, file_path):
    with io.open(file_path, mode="w", encoding="utf-8") as f:
        for sent in list_sent:
            f.write(sent + "\n")  # .encode("utf-8")

def save_to_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def undo_rephrase(lst):
    return " ".join(lst).replace("for_example", "for example").split()

def rephrase(str):
    return str.replace("for example", "for_example")

def get_books_pairs(file_path, start_index, end_index):
    # these words, when they appear sentence-initially,
    # are almost certainly discourse markers between the
    # previous sentence and this one
    clean_initial = [
        "also", "and", "but", "for example", "however", "meanwhile",
        "so", "then"
    ]

    # these words, when they appear sentence-internally,
    # are probably serving as discourse markers that split the
    # full sentence into two mini-sentences
    clean_internal = [
        "after", "although", "because", "before", "but", "for example",
        "if", "when", "while"
    ]

    discourse_markers = list(set(clean_internal + clean_initial))

    sents = {d: [] for d in discourse_markers}

    total_pairs_extracted = 0
    with io.open(file_path, 'rU', encoding="utf-8") as f:
        prev_words = None
        i = 0
        for line in f:
            i += 1
            if i < start_index:
                pass
            elif i > end_index:
                break
            sent = line[:-1]
            if i % 1000000 == 0:
                print("reading sentence {}".format(i))
            words = rephrase(sent).split()  # strip puncts and then split (already tokenized)
            # all of these have try statements, because sometimes the discourse marker will
            # only be a part of the word, and so it won't show up in the words list
            for marker in discourse_markers:
                if marker == "for example":
                    proxy_marker = "for_example"
                else:
                    proxy_marker = marker
                if proxy_marker in words[1:] and marker in clean_internal: # sentence-internal
                    idx = words.index(proxy_marker)
                    sents[marker].append((undo_rephrase(words[:idx]), undo_rephrase(words[idx+1:])))
                    total_pairs_extracted += 1
                elif marker in clean_initial and prev_words!=None and words[0].lower()==marker:
                    sents[marker].append((prev_words, undo_rephrase(words[1:])))
                    total_pairs_extracted += 1

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

    if args.aggregate:
        pairs = {}
        for file_path in glob.glob(pjoin(data_dir, "*_*-*.pkl")):
            print(file_path)
            file_data = pickle.load(open(file_path, "rb"))
            for key in file_data:
                if not key in pairs:
                    pairs[key] = []
                pairs[key] += file_data[key]

        n=0
        for key in pairs: n+=len(pairs[key])
        print("total pairs extracted: {}".format(n))

        for key in pairs: print("{} ~ {} ({}%)".format(
            key,
            len(pairs[key]),
            float(len(pairs[key]))/n*100
        ))

        pickle.dump(pairs, open(args.output_filename, "wb"))

    else:

        # directly use wikitext-103

        total = 40000000
        segment_length = total / round(args.n_segments)

        start_index = segment_length*args.segment_index
        end_index =  segment_length*(args.segment_index + 1)

        bookcorpus_path = pjoin("data", "books", "books_large_p1.txt")
        all_sentences_pairs_1 = get_books_pairs(bookcorpus_path, start_index, end_index)

        bookcorpus_path = pjoin("data", "books", "books_large_p2.txt")
        all_sentences_pairs_2 = get_books_pairs(bookcorpus_path, start_index, end_index)

        all_sentences_pairs = merge_dict(all_sentences_pairs_1, all_sentences_pairs_2)

        output_filename = "books_{}_{}.pkl".format(start_index, end_index)

        save_to_pickle(all_sentences_pairs, pjoin("data", "books", output_filename))
