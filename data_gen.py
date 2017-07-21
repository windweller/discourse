#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Corpus (e.g. PTB, Wikitext2) Pre-processing

- does the split (train, test, valid) if necessary
- tokenization if necessary

For each discourse marker:
  - make tuple for each instance of that marker IF
    it attaches to the head of the full sentence AND
    it matches our accepted depparse patterns
    (s1, s2, discourse marker)

Make files `S1.txt`, `S2.txt`, and `labels.txt`.

(Also make master files aggregated accross "all" corpora)
"""

import re
import io
import sys
import nltk
import os
from os.path import join as pjoin

import json
import pickle
import requests
import datetime

import logging
logging.basicConfig(level=logging.INFO)

import sys
reload(sys)
sys.setdefaultencoding('utf8')

rejection_mode = False


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

"""
Parse a particular corpus
tags implemented so far: "ptb" and "wiki" (wikitext-103)
"""
def main():
    # for each sentence,
    # regex to determine if depparse is necessary
    # (if so, depparse)
    # extract pairs for each matched marker
    print(datetime.datetime.now().time())
    parse_corpus("wiki")
    parse_corpus("ptb")


"""
Go through corpus files, extract sentence pairs, and write all the data
to a pickle file.

Format of pickle file:
{"discourse_marker": [["this", "is", "s1"], ["this", "is", "s2"]]}

Names of files for each corpus are hardcoded in here
"""
def parse_corpus(corpus):
    assert(corpus in ["wiki", "ptb"])

    # hardcoding of corpus filenames
    # need a list of filenames, they'll get mixed together
    if corpus == "ptb":
        data_dir = "data/ptb"
        corpus_files = [
            "ptb.valid.txt",
            "ptb.train.txt",
            "ptb.test.txt"
        ]
    elif corpus == "wiki":
        data_dir = "data/wikitext-103"
        corpus_files = [
            "wiki.valid.tokens",
            "wiki.train.tokens",
            "wiki.test.tokens"
        ]

    all_sentence_pairs, rejected_sentences = get_pairs_from_files(
        data_dir=data_dir,
        corpus_files=corpus_files
    )

    save_path = pjoin(data_dir, "all_sentence_pairs.pkl")
    pickle.dump(all_sentence_pairs, open(save_path, "wb"))
    pickle.dump(
        rejected_sentences,
        open(pjoin(data_dir, "rejected_sentences.pkl"), "wb")
    )


def get_pairs_from_files(data_dir, corpus_files):
    discourse_markers = [
        "because", "although",
        "but", "for example", "when",
        "before", "after", "however", "so", "still", "though",
        "meanwhile",
        "while", "if"
    ]
    rejected_sentences = {d: [] for d in discourse_markers}

    # for each input file
    for file_name in corpus_files:
        file_path = pjoin(data_dir, file_name)
        save_path = pjoin(data_dir, file_name + "_cache.pkl")
        if os.path.isfile(save_path):
            pairs, starting_line_num = pickle.load(open(save_path, "rb"))
        else:
            starting_line_num = -1
            pairs = {d: [] for d in discourse_markers}
        with open(file_path, 'r') as f:
            logging.info("loading file {}".format(file_path))
            # for each line
            line = f.readline()
            s = None
            line_num = 0
            while line:
                if line_num > starting_line_num:
                    if line_num % 1000 == 0:
                        logging.info("{} loading line {}".format(datetime.datetime.now().time(), line_num))
                        pickle.dump((pairs, line_num), open(save_path, "wb"))
                    words = line.split()
                    if len(words)==0 or words[0]=="=":
                        s = None
                    else:
                        # extract all sentences from that line
                        sentences = line.strip().split(" . ")
                        while len(sentences)>0:
                            # and for each sentence and its previous sentence
                            prev_s, s = s, sentences.pop()
                            # first determine via regex whether a depparse is necessary
                            markers, depparse_required = search_sentence_for_marker(s)
                            # (run depparse if necessary)
                            parse = None
                            if depparse_required:
                                parse = get_parse(s)
                            # then for each marker in the sentence,
                            found_pair = False
                            for marker in markers:
                                # add to that list of pairs
                                pair_for_marker = extract_pairs_from_sentence(
                                    current_sentence = s,
                                    marker = marker,
                                    previous_sentence = prev_s,
                                    parse = parse
                                )
                                if pair_for_marker:
                                    pairs[marker.lower()].append(pair_for_marker)
                                    found_pair = True
                            if not found_pair:
                                has_marker = re.match(".*(" + "|".join(discourse_markers) + ")", s)
                                if has_marker:
                                    for m in has_marker.groups():
                                        rejected_sentences[m.lower()].append(s)
                line = f.readline()
                line_num += 1
            pickle.dump((pairs, line_num), open(save_path, "wb"))
    return pairs, rejected_sentences


"""
Given a sentence (a string, space separated),
run a regex to see
   A) whether any of our discourse markers are in that sentence and 
   B) whether we'll need a dependency parse
"""
def search_sentence_for_marker(sentence_string):

    # initialize empty
    markers = []
    depparse_required = False

    # collapse "meanwhile" into a single word
    sentence_string = re.sub(
        "in the meanwhile",
        "meanwhile",
        sentence_string,
        re.IGNORECASE
    )

    # first, search for the simple examples
    # we never need a dependency parse for "but", "for example", or "when"
    simple = re.match(
        ".*(but|for example|when|meanwhile)",
        sentence_string,
        re.IGNORECASE
    )
    if simple:
        markers += simple.groups()

    # next, search for markers that, if they appear sentence-internally,
    # do not require a dependency parse
    sentence_internal = re.match(
        # reject "because of", "as if", "a while", and "all the while"
        ".* (because|although|if|while) ",
        sentence_string,
        re.IGNORECASE
    )
    # reject only if that was matched
    # fix me
    rejections = re.match(
        ".*(because of|as if|a while|all the while)",
        sentence_string,
        re.IGNORECASE
    )
    if sentence_internal and (not rejection_mode or not rejections):
        markers += sentence_internal.groups()

    # next search for those same markers sentence-initially
    # (this will require a depparse)
    sentence_initial = re.match(
        "^(because|although|if|while) ",
        sentence_string,
        re.IGNORECASE
    )
    if sentence_initial:
        markers += sentence_initial.groups()
        depparse_required = True

    # next, look for markers that we're positively pattern-matching
    # (the ones that are too variable to split without a dependency parse)
    positive_pattern_match = re.match(
        ".*(before|after|however|so|still|though)",
        sentence_string,
        re.IGNORECASE
    )
    if positive_pattern_match:
        markers += positive_pattern_match.groups()
        depparse_required = True

    return (markers, depparse_required)


"""
POS-tag string as if it's a sentence
and see if it has a verb that could plausibly be the predicate.
"""
# fix me
def has_verb(string):
    if rejection_mode:
        parse = get_parse(string, depparse=False)
        tokens = json.loads(parse)["sentences"][0]["tokens"]
        return any([re.match("V(?!(BG|BN))", t["pos"]) for t in tokens])
    else:
        return True


"""
Given a sentence (and possibly its parse and previous sentence)
and a particular discourse marker,
find all valid S1, S2 pairs for that discourse marker
"""
def extract_pairs_from_sentence(current_sentence, marker, previous_sentence = "", parse = None):

    S1, S2 = search_for_S1_S2_candidates(
        current_sentence = current_sentence,
        marker = marker, 
        previous_sentence = previous_sentence,
        parse = parse
    )
    if S1 and S2:
        # check that both s1 and s2 have verbs that could be the predicate
        S1_valid = has_verb(S1)
        S2_valid = has_verb(S2)
        if S1_valid and S2_valid:
            return (S1, S2)
        else:
            if S2_valid and marker=="meanwhile":
                S1 = previous_sentence
                S2 = re.sub(" ?meanwhile ?", " ", current_sentence).strip()
                return (S1, S2)
            else:
                return None
    if rejection_mode:
        return None
    else:
        m = re.match("(.*)" + marker + "(.*)", current_sentence)
        groups = m.groups()
        if len(groups)==1:
            return (previous_sentence, groups[0])
        elif len(groups)==2:
            return (groups[0], groups[1])
        else:
            return None


"""
Use the rules we came up with (sometimes simple regex rules, sometimes
depparse searches) for each discourse marker to pull out a candidate
S1 and S2
"""
def search_for_S1_S2_candidates(current_sentence, marker, previous_sentence = "", parse = None):

    # first, look for the simplest discourse markers, that we can
    # handle with just regexes
    if marker in ["but", "for example", "when", "meanwhile"]:
        simple = re.match(
            "(.*)(?:but|for example|when|meanwhile)(.*)",
            current_sentence,
            re.IGNORECASE
        )
        simple_groups = simple.groups()
        if len(simple_groups)==1:
            s1 = previous_sentence
            s2 = simple_groups[0].strip()
            return (s1, s2)
        else:
            s1 = simple_groups[0].strip()
            s2 = simple_groups[1].strip()
            return (s1, s2)

    # then, look at the discourse markers that we can handle with
    # regexes when they're sentence internal
    if marker in ["because", "although", "while", "if"]:
        # first handle them if they're sentence-internal
        sentence_internal = re.match(
            "(.+) (because|although|while|if) (.+)",
            current_sentence,
            re.IGNORECASE
        )
        if sentence_internal:
            s1 = sentence_internal.groups()[0]
            s2 = sentence_internal.groups()[2]
            return (s1, s2)
        else:
            # otherwise handle them if they're sentence-initial
            search_results_for_S1_s1 = search_for_S2_S1_order(
                marker,
                parse,
                previous_sentence
            )
            if search_results_for_S1_s1:
                s1, s2 = search_results_for_S1_s1
                return (s1, s2)
            else:
                if marker=="if":
                    # reject pattern: "S1. If S2."
                    return (None, None)
                else:
                    # accept pattern: "S1. [discourse marker] S2."
                    # (for all but "if")
                    s1 = previous_sentence
                    s2 = re.sub(
                        "^" + marker + " ",
                        "",
                        current_sentence,
                        re.IGNORECASE
                    )
                    return (s1, s2)

    # finally, look for markers that we're positively pattern-matching
    # (the ones that are too variable to split without a dependency parse)
    if marker in ["before", "after", "however", "so", "still", "though"]:
        search_results = search_for_dep_pattern(
            marker=marker,
            current_sentence=current_sentence,
            previous_sentence=previous_sentence,
            parse_string=parse
        )
        if search_results:
            return search_results
        else:
            return (None, None)

    return (None, None)

"""
using the depparse, look for the desired pattern, in any order
"""
def search_for_dep_pattern(marker, current_sentence, previous_sentence, parse_string):  
    parse = json.loads(parse_string)
    sentence = Sentence(parse["sentences"][0])
    return sentence.find_pair(marker, "any", previous_sentence)


"""
using the depparse, look for the "reverse" order
e.g. "If A, B." (as opposed to "B if A.")
"""
def search_for_S2_S1_order(marker, parse_string, previous_sentence):
    parse = json.loads(parse_string)
    sentence = Sentence(parse["sentences"][0])
    return sentence.find_pair(marker, "s2 discourse_marker s1", previous_sentence)


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


"""
use corenlp server (see https://github.com/erindb/corenlp-ec2-startup)
to parse sentences: tokens, dependency parse
"""
def get_parse(sentence, depparse=True):
    # fix me: undo this retokenization later!
    sentence = re.sub(" 't ", "'t ", sentence)
    if depparse:
        url = "http://localhost:12345?properties={annotators:'tokenize,ssplit,pos,depparse'}"
    else:
        url = "http://localhost:12345?properties={annotators:'tokenize,ssplit,pos'}"
    data = sentence
    parse_string = requests.post(url, data=data).text
    return parse_string


class Sentence():
    def __init__(self, json_sentence):
        self.json = json_sentence
        self.dependencies = json_sentence["basicDependencies"]
        self.tokens = json_sentence["tokens"]
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
    def string(self):
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
        # fix me

        # given an index,
        # grab every word that's a child of it in the dependency graph
        subordinates = self.get_subordinate_indices(
        acc=[head_index],
        explore=[head_index],
        exclude_indices=exclude_indices
        )
        subordinates.sort()

        subordinate_phrase = " ".join([self.word(i) for i in subordinates])

        # optionally exclude some indices and their children

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
                s2_ind = s2_head_index
                for s1_head_index in self.get_candidate_S1_indices(marker, s2_head_index):
                    # store S1 if we have one
                    S1 = self.get_phrase_from_head(
                        s1_head_index,
                        exclude_indices=[s2_head_index]
                    )
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
