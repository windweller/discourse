#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Corpus (e.g. PTB, Wikitext2) Pre-processing

For each discourse marker, make a tuple (s1, s2) where
s1 and s2 are extracted according to some common patterns.
Patterns are specific to each discourse markers (though there
are groups of discourse markers that use the same pattern).
Sometimes we run a dependency parse on the sentence,
other times we use simple sentence-splitting rules.

Make files:
* `[raw filename]_[start index]-[end index].pkl`
    - contains all extracted pairs from sentences with
      index in [`start index`, `end index`) within that file
    - format:
        ```
        {
            "discourse marker": [
                (["tokens", "in", "s1"], ["tokens", "in", "s2"]),
                ...
            ],
            ...
        }
        ```
* `[raw filename].SENTENCES`.

This will later be aggregated into a single `all_sentence_pairs.pkl` file.
"""

import io
import nltk
import os
from os.path import join as pjoin

import json
import pickle
import requests
import datetime
import argparse
import numpy as np

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import nltk
import glob


"""
Positive dependency parse patterns we're looking for

S1 - [list] a set of accepted dependencies from
   the first to second sentence
   S1 head ---> S2 head (full S head)

S2 - [string] the single kind of dependency
   for the marker to have with the second sentence
   S2 head (full S head) ---> connective

POS - [string] accepted part of speech for marker
    (not using this yet: fix me!)
"""
dependency_patterns = {
  "after": {
    "POS": "IN",
    "S2": "mark", # S2 head (full S head) ---> connective
    "S1": ["advcl"]
  },
  "although": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  },
  "before": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  },
  "so": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  },
  "still": {
    "POS": "RB",
    "S2": "advmod",
    "S1": ["parataxis", "dep"]
  },
  "though": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  },
  "because": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  },
  "however": {
    "POS": "RB",
    "S2": "advmod",
    "S1": ["dep", "parataxis"]
  },
  "if": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  },
  "while": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  }
}

discourse_markers = [
    "because", "although",
    "but", "for example", "when",
    "before", "after", "however", "so", "still", "though",
    "meanwhile",
    "while", "if"
]

def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument("--data_dir", default="data/wikitext-103")
    parser.add_argument("--corpus_files", default="wiki.valid.tokens wiki.test.tokens wiki.train.tokens")
    parser.add_argument("--corpus_length", default=5666000, type=int)
    parser.add_argument("--n_cores", default=4, type=int)
    parser.add_argument("--segment_index", default=0, type=int)
    parser.add_argument("--shortest_sentence_length", default=10, type=int)
    parser.add_argument("--mode", default="process_raw")
    parser.add_argument("--sentences_presegmented", action='store_true')
    parser.add_argument("--starting_sentence_index", default=None, type=int)
    parser.add_argument("--ending_sentence_index", default=None, type=int)
    return parser.parse_args()

"""
Parse a particular corpus
tags implemented so far: "ptb" and "wiki" (wikitext-103)
"""
def main():
    args = setup_args()
    if args.mode == "process_raw":
        process_raw_files(args)
    elif args.mode == "aggregate":
        aggregate_prcessed_files(args)


def process_raw_files(args):

    # for each sentence,
    # regex to determine if depparse is necessary
    # (if so, depparse)
    # extract pairs for each matched marker
    print(datetime.datetime.now().time())
    # parse_corpus("wiki")
    # parse_corpus("ptb")

    # wikitext parameters:
    data_dir = args.data_dir
    corpus_files = args.corpus_files.split()
    corpus_length = args.corpus_length #approx (actual value is a little bit less)

    hard_indices_given = args.starting_sentence_index!=None or args.ending_sentence_index!=None

    if hard_indices_given:
        if args.starting_sentence_index:
            starting_sentence_index = args.starting_sentence_index
        else:
            starting_sentence_index = 0
        if args.ending_sentence_index:
            ending_sentence_index = args.ending_sentence_index
        else:
            ending_sentence_index = corpus_length
    else:
        n_cores = args.n_cores

        step = int(np.ceil(corpus_length / n_cores))
        starting_indices = [i for i in range(0, corpus_length, step)]
        ending_indices = [i+step for i in starting_indices]

        segment_index = args.segment_index
        starting_sentence_index = starting_indices[segment_index]
        ending_sentence_index = ending_indices[segment_index]

    print("parsing from {}K to {}K...".format(
        starting_sentence_index/1000,
        ending_sentence_index/1000
    ))

    for filename in corpus_files:
        save_path = pjoin(
            data_dir, 
            "{}_{}-{}.pkl".format(filename, starting_sentence_index, ending_sentence_index)
        )

        if os.path.isfile(save_path):
            print("file {} already exists".format(save_path))
        else:
            print("processing file {}...".format(filename))
            pairs_from_split = get_wiki_pairs(
                pjoin(data_dir, filename),
                starting_sentence_index,
                ending_sentence_index,
                args.shortest_sentence_length,
                args.sentences_presegmented
            )
            pickle.dump(pairs_from_split, open(save_path, "wb"))


def aggregate_prcessed_files(args):
    data_dir = args.data_dir
    pairs = {d: [] for d in discourse_markers}
    for file_path in glob.glob(pjoin(data_dir, "*_*-*.pkl")):
        print(file_path)
        file_data = pickle.load(open(file_path, "rb"))
        for key in pairs:
            pairs[key] += file_data[key]

    n=0
    for key in pairs: n+=len(pairs[key])
    print("total pairs extracted: {}".format(n))

    for key in pairs: print("{} ~ {} ({}%)".format(
        key,
        len(pairs[key]),
        float(len(pairs[key]))/n*100
    ))

    pickle.dump(pairs, open("data/wikitext-103/all_sentence_pairs.pkl", "wb"))



def get_wiki_pairs(file_path, starting_sentence_index, ending_sentence_index, shortest_sentence_length, sentences_presegmented):
    pairs = {d: [] for d in discourse_markers}

    with io.open(file_path, 'rU', encoding="utf-8") as f:
        if sentences_presegmented:
            print("reading sentences...")
            # read and store only the lines we need to
            sent_list = []
            line_num = 0
            for line in f:
                if line_num < starting_sentence_index:
                    pass
                elif ending_sentence_index <= line_num:
                    break
                else:
                    sent_list.append(line[:-1])
                line_num += 1
        else:
            tokens = f.read().replace("\n", ". ")

            save_path = file_path + ".SENTENCES"
            if os.path.isfile(save_path):
                print("loading sentences...")
                sent_list = pickle.load(open(save_path, "rb"))
                print("sentences loaded. total number of sentences: " + str(len(sent_list)))
            else:
                print("tokenizing...")
                sent_list = nltk.sent_tokenize(tokens)
                pickle.dump(sent_list, open(save_path, "wb"))
                print("tokenization complete. total number of sentences: " + str(len(sent_list)))
 
        previous_sentence = ""

        if sentences_presegmented:
            sent_num = starting_sentence_index
        else:
            sent_num = 0
            
        for sent in sent_list:
            if ending_sentence_index <= sent_num:
                break
            if sent_num % 1000 == 0:
                print("{} - {}".format(
                    datetime.datetime.now().time(),
                    sent_num
                ))
            if sent_num >= starting_sentence_index:
                for marker in discourse_markers:
                    if sent.lower().find(marker) >= 0:

                        if marker == "for example":
                            if sent.find("for example") >=0:
                                sent = sent.replace("for example", "for_example")
                            elif sent.find("For example") >=0:
                                sent = sent.replace("For example", "for_example")
                            marker = "for_example"

                        search_words = sent.lower().split()

                        if marker in search_words:
                            pair = get_pairs_from_sentence(
                                sent,
                                marker,
                                previous_sentence
                            )
                            if pair and all([len(p)>shortest_sentence_length for p in pair]):
                                if marker == "for_example":
                                    marker = "for example"
                                pairs[marker].append((pair[0].split(), pair[1].split()))
                previous_sentence = sent
            sent_num += 1
    return pairs


def get_pairs_from_sentence(sent, marker, previous_sentence):

    if marker == "meanwhile":
        sent = sent.replace("in the meanwhile", "meanwhile")
    elif marker == "while":
        if " a while " in sent or " all the while " in sent:
            return None
    elif marker == "because":
        if " because of " in sent or "Because of" in sent:
            return None

    words = sent.split()
    # 1. decide whether dependency parsing is needed
    words_to_search = sent.lower().split()
    marker_index = words_to_search.index(marker)

    # first, look for the simplest discourse markers, that we can
    # handle with just regexes
    # and handle them
    if marker in ["but", "for_example", "when", "meanwhile"]:
        if marker_index == 0:
            return (previous_sentence, " ".join(words[1:]))
        else:
            return (
                " ".join(words[0:marker_index]),
                " ".join(words[marker_index+1:len(words)])
            )

    # then, look at the discourse markers that we can handle with
    # regexes when they're sentence internal
    elif marker in ["because", "although", "while", "if"]:
        if marker_index == 0:
            # handle them if they're sentence-initial

            # search for pattern:
            # "[discourse marker] S2, S1" (needs dependency parse)
            reverse_pattern_pair = search_for_reverse_pattern_pair(sent, marker, words, previous_sentence)

            if reverse_pattern_pair:
                # if we find it, return it.
                return reverse_pattern_pair
            else:
                # otherwise, return pattern: "S1. [discourse marker] S2."
                return (previous_sentence, " ".join(words[1:]))
        else:
            return (
                " ".join(words[0:marker_index]),
                " ".join(words[marker_index+1:len(words)])
            )

    # finally, look for markers that we're positively pattern-matching
    # (the ones that are too variable to split without a dependency parse)
    elif marker in ["before", "after", "however", "so", "still", "though"]:
        return search_for_dep_pattern(
            marker=marker,
            current_sentence=sent,
            previous_sentence=previous_sentence
        )

    else:
        raise Exception("error in marker comparison")


# search for pattern:
# "[discourse marker] S2, S1" (needs dependency parse)
def search_for_reverse_pattern_pair(sent, marker, words, previous_sentence):
    parse_string = get_parse(sent, depparse=True)

    # book corpus maybe has carriage returns and new lines and other things?
    try: 
        parse = json.loads(parse_string.replace('\r\n', ''))
    except ValueError:
        parse = json.loads(re.sub("[^A-z0-9.,!?\"'*&/\{\}\[\]()=+-]", "", parse_string))
        
    sentence = Sentence(parse["sentences"][0], sent)
    return sentence.find_pair(marker, "s2 discourse_marker s1", previous_sentence)


def is_verb_tag(tag):
    return tag[0] == "V" and not tag[-2:] in ["BG", "BN"]


"""
POS-tag string as if it's a sentence
and see if it has a verb that could plausibly be the predicate.
"""
def has_verb(string):
    parse = get_parse(string, depparse=False)
    tokens = json.loads(parse)["sentences"][0]["tokens"]
    return any([is_verb_tag(t["pos"]) for t in tokens])


"""
using the depparse, look for the desired pattern, in any order
"""
def search_for_dep_pattern(marker, current_sentence, previous_sentence):  
    parse_string = get_parse(current_sentence, depparse=True)

    # book corpus maybe has carriage returns and new lines and other things?
    try: 
        parse = json.loads(parse_string.replace('\r\n', ''))
    except ValueError:
        parse = json.loads(re.sub("[^A-z0-9.,!?\"'*&/\{\}\[\]()=+-]", "", parse_string))

    sentence = Sentence(parse["sentences"][0], current_sentence)
    return sentence.find_pair(marker, "any", previous_sentence)


# https://stackoverflow.com/a/18669080
def get_indices(lst, element):
  result = []
  offset = -1
  while True:
    try:
      offset = lst.index(element, offset+1)
    except ValueError:
      return result
    result.append(offset)

def get_nearest(lst, element):
    distances = [abs(e-element) for e in lst]
    return lst[np.argmin(distances)]


"""
parsed tokenization is different from original tokenization.
try to re-align and extract the correct words given the
extraction_indices (which are 1-indexed into parsed_words)

fix me to catch more cases?
"""
def extract_subphrase(orig_words, parsed_words, extraction_indices):
    extraction_indices = [i-1 for i in extraction_indices]

    if len(orig_words) == len(parsed_words):
        return " ".join([orig_words[i] for i in extraction_indices])
    else:
        first_parse_index = extraction_indices[0]
        first_word_indices = get_indices(orig_words, parsed_words[first_parse_index])

        last_parse_index = extraction_indices[-1]

        last_word_indices = get_indices(orig_words, parsed_words[last_parse_index])

        if len(first_word_indices)>0 and len(last_word_indices)>0:
            first_orig_index = get_nearest(first_word_indices, first_parse_index)
            last_orig_index = get_nearest(last_word_indices, last_parse_index)
            if last_orig_index-first_orig_index == last_parse_index-first_parse_index:
                # maybe it's just shifted
                shift = first_orig_index - first_parse_index
                extraction_indices = [i+shift for i in extraction_indices]
                return " ".join([orig_words[i] for i in extraction_indices])
            else:
                # or maybe there's funny stuff happening inside the subphrase
                # in which case T-T
                return None
        else:
            if len(first_word_indices)>0 and abs(last_parse_index-len(parsed_words))<3:
                # the end of the sentence is always weird. assume it's aligned

                # grab the start of the subphrase
                first_orig_index = get_nearest(first_word_indices, first_parse_index)
                # shift if necessary
                shift = first_orig_index - first_parse_index
                extraction_indices = [i+shift for i in extraction_indices]

                if len(orig_words) > extraction_indices[-1]:
                    # extend to the end of the sentence if we're not already there
                    extraction_indices += range(extraction_indices[-1]+1, len(orig_words))
                else:
                    extraction_indices = [i for i in extraction_indices if i<len(orig_words)]

                return " ".join([orig_words[i] for i in extraction_indices])

            else:
                # or maybe the first and/or last words have been transformed,
                # in which case T-T
                return None
        


"""
use corenlp server (see https://github.com/erindb/corenlp-ec2-startup)
to parse sentences: tokens, dependency parse
"""
def get_parse(sentence, depparse=True):
    sentence = sentence.replace("'t ", " 't ")
    if depparse:
        url = "http://localhost:12345?properties={annotators:'tokenize,ssplit,pos,depparse'}"
    else:
        url = "http://localhost:12345?properties={annotators:'tokenize,ssplit,pos'}"
    data = sentence
    parse_string = requests.post(url, data=data).text
    return parse_string


class Sentence():
    def __init__(self, json_sentence, original_sentence):
        self.json = json_sentence
        self.dependencies = json_sentence["basicDependencies"]
        self.tokens = json_sentence["tokens"]
        self.original_sentence = original_sentence
    def indices(self, word):
        if len(word.split(" ")) > 1:
            words = word.split(" ")
            indices = [i for lst in [self.indices(w) for w in words] for i in lst]
            return indices
        else:
            return [i+1 for i in get_indices([t["word"] for t in self.tokens], word)]
    def token(self, index):
        return self.tokens[index-1]
    def word(self, index):
        return self.token(index)["word"]
    def find_parents(self, index, filter_types=False):
        deps = self.find_deps(index, dir="parents", filter_types=filter_types)
        return [d["governor"] for d in deps]
    def find_children(self, index, filter_types=False):
        deps = self.find_deps(index, dir="children", filter_types=filter_types)
        return [d["dependent"] for d in deps]
    def find_deps(self, index, dir=None, filter_types=False):
        deps = []
        if dir=="parents" or dir==None:
            deps += [d for d in self.dependencies if d['dependent']==index]
        if dir=="children" or dir==None:
            deps += [d for d in self.dependencies if d['governor']==index]
        if filter_types:
            return [d for d in deps if d["dep"] in filter_types]
        else:
            return deps
    def find_dep_types(self, index, dir=None, filter_types=False):
        deps = self.find_deps(index, dir=dir, filter_types=filter_types)
        return [d["dep"] for d in deps]
    def __str__(self):
        return " ".join([t["word"] for t in self.tokens])
    def get_subordinate_indices(self, acc, explore, exclude_indices=[]):
        children = [c for i in explore for c in self.find_children(i) if not c in exclude_indices]
        if len(children)==0:
            return acc
        else:
            return self.get_subordinate_indices(
                acc=acc + children,
                explore=children,
                exclude_indices=exclude_indices
            )

    def get_phrase_from_head(self, head_index, exclude_indices=[]):

        # given an index,
        # grab every index that's a child of it in the dependency graph
        subordinate_indices = self.get_subordinate_indices(
            acc=[head_index],
            explore=[head_index],
            exclude_indices=exclude_indices
        )
        subordinate_indices.sort()

        # make string of subordinate phrase from parse
        parse_subordinate_string = " ".join([self.word(i) for i in subordinate_indices])

        # correct subordinate phrase from parsed version to wikitext version
        # (tokenization systems are different)
        orig_words = self.original_sentence.split()
        parsed_words = [t["word"] for t in self.tokens]

        subordinate_phrase = extract_subphrase(orig_words, parsed_words, subordinate_indices)

        # make a string from this to return
        return subordinate_phrase

    def get_valid_marker_indices(self, marker):
        pos = dependency_patterns[marker]["POS"]
        return [i for i in self.indices(marker) if pos == self.token(i)["pos"] ]

    def get_candidate_S2_indices(self, marker, marker_index):
        connection_type = dependency_patterns[marker]["S2"]
        # Look for S2
        return self.find_parents(marker_index, filter_types=[connection_type])

    def get_candidate_S1_indices(self, marker, s2_head_index):
        valid_connection_types = dependency_patterns[marker]["S1"]
        return self.find_parents(
            s2_head_index,
            filter_types=valid_connection_types
        )

    def find_pair(self, marker, order, previous_sentence):
        assert(order in ["s2 discourse_marker s1", "any"])
        # fix me
        # (this won't quite work if there are multiple matching connections)
        # (which maybe never happens)
        S1 = None
        S2 = None
        s1_ind = 1000
        s2_ind = 0

        for marker_index in self.get_valid_marker_indices(marker):
            for s2_head_index in self.get_candidate_S2_indices(marker, marker_index):
                # store S2 if we have one
                S2 = self.get_phrase_from_head(
                    s2_head_index,
                    exclude_indices=[marker_index]
                )
                # we'll lose some stuff here because of alignment between
                # wikitext tokenization and corenlp tokenization.
                # if we can't get a phrase, reject this pair
                if not S2:
                    return None
                s2_ind = s2_head_index
                for s1_head_index in self.get_candidate_S1_indices(marker, s2_head_index):
                    # store S1 if we have one
                    S1 = self.get_phrase_from_head(
                        s1_head_index,
                        exclude_indices=[s2_head_index]
                    )
                    # we'll lose some stuff here because of alignment between
                    # wikitext tokenization and corenlp tokenization.
                    # if we can't get a phrase, reject this pair
                    if not S1:
                        return None
                    s1_ind = s1_head_index

        # if we are only checking for the "reverse" order, reject anything else
        if order=="s2 discourse_marker s1":
            if s1_ind < s2_ind:
                return None

        if S2 and not S1:
            S1 = previous_sentence

        if S1 and S2:
            return S1, S2
        else:
            return None


if __name__ == '__main__':
  main()
