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

import logging
logging.basicConfig(level=logging.INFO)

import sys
reload(sys)
sys.setdefaultencoding('utf8')


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
dependency_types = {
  "after": {
     # this will collect some verbal noun phrases
     # if we want to exclude them, we can on the basis of the head being a VBG
    "POS": "IN",
    "S2": "mark", # S2 head (full S head) ---> connective
    "S1": ["advcl"]#, # S2 head (full S head) ---> S1 head
    # # fix me! (i'm not using alternates and lost alternates)
    # "alternates": ["and after"],
    # "lost_alternates": ["after that"]
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
    "S2": "mark", # S2 head (full S head) ---> connective
    "S1": ["advcl"], # S1 head ---> S2 head (full S head)
    # "alternates": ["just because", "only because"], # fix me (????)
    # "lost_alternates": ["because of"]
  },
  "however": {
    # "however you interpret it, the claims are wrong"
    # uses advcl for S2, so we're resolving that ambiguity
    "POS": "RB",
    "S2": "advmod",
    # different kinds of possible dependencies for S1
    "S1": ["dep", "parataxis"]
  },
  "if": {
    # e.g. "it will fall if you rest it on the table like that"
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  },
  "while": {
    # e.g. "while he watched tv, he kept knitting the sweater"
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
        corpus_files = ["ptb.valid.txt", "ptb.train.txt", "ptb.test.txt"]
    elif corpus == "wiki":
        data_dir = "data/wikitext-103"
        corpus_files = [
            "wiki.valid.tokens",
            "wiki.train.tokens",
            "wiki.test.tokens"
        ]

    all_sentence_pairs = get_pairs_from_files(
        data_dir=data_dir,
        corpus_files=corpus_files
    )

    save_path = pjoin(data_dir, "all_sentence_pairs.pkl")
    pickle.dump(all_sentence_pairs, open(save_path, "wb"))


def get_pairs_from_files(data_dir, corpus_files):
    discourse_markers = [
        "because", "although",
        "but", "for example", "when",
        "before", "after", "however", "so", "still", "though",
        "meanwhile",
        "while", "if"
    ]
    pairs = {d: [] for d in discourse_markers}

    # for each input file
    for file_name in corpus_files:
        file_path = pjoin(data_dir, file_name)
        with open(file_path, 'r') as f:
            logging.info("loading file {}".format(file_path))
            # for each line
            line = f.readline()
            s = None
            while line:
                line = f.readline()
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
    return pairs


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
        "(but|for example|when|meanwhile)",
        sentence_string,
        re.IGNORECASE
    )
    if simple:
        markers += simple.groups()

    # next, search for markers that, if they appear sentence-internally,
    # do not require a dependency parse
    sentence_internal = re.match(
        # reject "because of", "as if", "a while", and "all the while"
        " (because(?! of)|although|(?!as )if|(?! a |all the )while) ",
        sentence_string,
        re.IGNORECASE
    )
    if sentence_internal:
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
        "(before|after|however|so|still|though)",
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
def has_verb(string):
    parse = get_parse(string, depparse=False)
    tokens = json.loads(parse)["sentences"][0]["tokens"]
    return any([re.match("V(?!(BG|BN))", t["pos"]) for t in tokens])


"""
Given a sentence (and possibly its parse and previous sentence)
and a particular discourse marker,
find all valid S1, S2 pairs for that discourse marker
"""
def extract_pairs_from_sentence(
            current_sentence,
            marker, 
            previous_sentence = "",
            parse = None):

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
    
    return None


"""
Use the rules we came up with (sometimes simple regex rules, sometimes
depparse searches) for each discourse marker to pull out a candidate
S1 and S2
"""
def search_for_S1_S2_candidates(
                current_sentence,
                marker, 
                previous_sentence = "",
                parse = None):

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
            "(.+) (?:because|although|while|if) (.+)",
            current_sentence,
            re.IGNORECASE
        )
        if sentence_internal:
            s1 = sentence_internal.groups()[0]
            s2 = sentence_internal.groups()[1]
            return (s1, s2)
        else:
            # otherwise handle them if they're sentence-initial
            search_results_for_S1_s1 = search_for_S2_S1_order(
                marker,
                parse
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
    # fix me

    return (None, None)


"""
using the depparse, look for the "reverse" order
e.g. "If A, B." (as opposed to "B if A.")
"""
def search_for_S2_S1_order(marker, parse):
    # fix me
    None


"""
use corenlp server (see https://github.com/erindb/corenlp-ec2-startup)
to parse sentences: tokens, dependency parse
"""
def get_parse(sentence, depparse=True):
    sentence = re.sub(" 't ", "'t ", sentence)
    if depparse:
        url = "http://localhost:12345?properties={annotators:'tokenize,ssplit,pos,depparse'}"
    else:
        url = "http://localhost:12345?properties={annotators:'tokenize,ssplit,pos'}"
    data = sentence
    parse_string = requests.post(url, data=data).text
    return parse_string






























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



discourse_markers = dependency_types.keys()


"""
data_dir - dir to save text files
split - valid, train, test
element_name - which variable: S1, S2, or labels
element_list - list of values for that variable
"""
def write_example_file(data_dir, split, element_name, element_list):
  savepath = os.path.join(data_dir, split + "_" + element_name + ".txt")
  w = open(savepath, mode="w")
  w.write("\n".join(element_list))
  w.close()


"""
marker - which discourse marker are we finding a pair for?
parse_string - full corenlp parse in json
current_sentence - full sentence
previous_sentence - previous sentence might be S1 for some sentences
"""
def parse_example(marker, parse_string, current_sentence, previous_sentence):

  # fix me (take sentence as input)
  # sentence = "i like her because she is nice"

  parse = json.loads(parse_string)

  # print(current_sentence)

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
        deps += [d for d in sentence.dependencies if d['dependent']==index]
      if dir=="children" or dir==None:
        deps += [d for d in sentence.dependencies if d['governor']==index]
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

      children = [c for i in explore for c in sentence.find_children(i) if not c in exclude_indices]
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

      subordinate_phrase = " ".join([sentence.word(i) for i in subordinates])

      # optionally exclude some indices and their children
      
      # make a string from this to return
      return subordinate_phrase

  sentence = Sentence(parse["sentences"][0])

  # marker_index = sentence.indices(marker)[0] # fix me!!!
  dep_patterns = dependency_types[marker]


  # Look for S2
  possible_marker_indices = sentence.indices(marker)
  possible_s2_head_indices = [p for marker_index in possible_marker_indices for p in sentence.find_parents(
    marker_index,
    filter_types=[dep_patterns["S2"]]
  )]
  # print(sentence.find_deps(possible_marker_indices[0]))
  if len(possible_s2_head_indices)==1:
    # Record S2
    s2_head_index = possible_s2_head_indices[0]
    s2 = sentence.get_phrase_from_head(
      s2_head_index,
      exclude_indices=[marker_index]
    )

    # Look for S1
    possible_s1_dependency_types = sentence.find_dep_types(
      s2_head_index,
      dir="parents"
    )
    if any([t in possible_s1_dependency_types for t in dep_patterns["S1"]]):
      possible_s1_head_indices = sentence.find_parents(
        s2_head_index,
        filter_types=dep_patterns["S1"]
      )
      if len(possible_s1_head_indices)==1:
        # if S1 found, record S1
        s1_head_index = possible_s1_head_indices[0]
        s1 = sentence.get_phrase_from_head(
          s1_head_index,
          exclude_indices=[s2_head_index]
        )
        return (s1, s2, marker)
      else:
        # if no S1 found, record previous sentence as S1
        s1 = previous_sentence
    else:
      # if no S1 found, record previous sentence as S1
      s1 = previous_sentence

    # if S2 found, return example tuple
    return (s1, s2, marker)

  else:
    # if no S2 found, print out sentence and return None
    return None



def text_filenames(data_dir, extension):
  return [os.path.join(data_dir, split + "_" + label + "." + extension) \
                      for label in ["S1", "S2", "labels"] \
                      for split in ["train", "valid", "test"]]

"""
data_tag - ptb or wikitext-103
collapse_nums - if we want, we can collapse numbers to N
strip_punctuation - if we want, we can strip punctuation
"""
def parse_data_directory(data_tag, collapse_nums=False, strip_punctuation=False):
  # Given a corpus where each line is a sentence, find examples with given
  # discourse markers.

  """
  PTB reduces all instances of numbers to N
  If we want to, we can do that for other corpora
  (By default, we don't)
  """
  def collapse_numbers(s):
    if COLLAPSE_NUMBERS:
      return re.sub("([^ ]*)\d([^ ]*)", "\1N\2", s)
    else:
      return s


  data_dir = "data/" + data_tag + "/"

  if data_tag == "wikitext-103":
    tag2 = "wiki"
    extension = "tokens"
  else:
    tag2 = data_tag
    extension = "txt"

  num_sentences = 0

  unparsed = []

  # download ptb if it doesn't exist
  # fix me
  if not os.path.isfile(data_dir + tag2 + ".test." + extension):
    # download it
    None
  if not os.path.isfile(data_dir + tag2 + ".train." + extension):
    None
  if not os.path.isfile(data_dir + tag2 + ".valid." + extension):
    None


  # check if text files already exist
  if False:#all([os.path.isfile(f) for f in text_filenames(data_dir, extension)]):
    print("text files in " + data_dir + " already exist. " +
          "Delete them if you want to rerun.")
  else:
    for split in ["train", "valid", "test"]:

      s1s = []
      s2s = []
      labels = []

      datapath = data_dir + tag2 + "." + split + "." + extension
      fd = open(datapath)
      line = fd.readline()
      previous_line = ""
      while line:
        line = re.sub("\n", "", line)
        line_words = line.strip().split()
        if len(line_words) > 0:
          if data_dir=="data/wikitext-103/" and line_words[0] == "=":
            previous_line = ""
          else:
            if data_dir=="data/wikitext-103/":
              doc_sentences = line.split(" . ")
            else:
              doc_sentences = [line]
            for s in doc_sentences:
              s = s.strip()
              if collapse_nums:
                words = collapse_numbers(s)

              if strip_punctuation:
                words_to_exclude = ["@-@", ".", "\"", ",", ":",
                                    "—", "(", ")", "@,@", "@.@",
                                    ";", "'", "–", "!", "?"]
                s = " ".join([w for w in s.split() if not w in words_to_exclude])

              pattern = ".* (then) .*"
              # pattern = ".* (" + "|".join(discourse_markers) + ") .*"
              m = re.match(pattern, s)
              if m:
                s = re.sub("[Ii]n the meanwhile ", "meanwhile ", s)
                parse_string = get_parse(s)
                for marker in m.groups():
                  if marker=="if":
                    if re.match("as if", s):
                      unparsed.append((s, marker))
                      print("NO MATCH: " + s)
                      continue
                  if marker=="while":
                    if re.match("( a|all the) while", s):
                      unparsed.append((s, marker))
                      print("NO MATCH: " + s)
                      continue
                  example_tuple = parse_example(marker, parse_string, s, previous_line)
                  if example_tuple:
                    s1, s2, label = example_tuple
                    s1s.append(s1.strip())
                    s2s.append(s2.strip())
                    labels.append(marker)
                  else:
                    unparsed.append((s, marker))
                    print("NO MATCH: " + s)
              previous_line = s

        previous_line = line
        line = fd.readline()
        num_sentences += 1
      fd.close()

      write_example_file(data_dir, split, "S1", s1s)
      write_example_file(data_dir, split, "S2", s2s)
      write_example_file(data_dir, split, "labels", labels)


if __name__ == '__main__':
  main()
