#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Corpus (PTB/Wikitext2) Pre-processing

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

import sys
reload(sys)
sys.setdefaultencoding('utf8')


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
Dependency parse patterns we're looking for

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
  "also": {
    # e.g. "i also like ice cream"
    #      "i like ice cream also"
    # definitely want sentence initial
    # fix me!!!
    # if "also" appears with other discourse markers, exclude it
    "POS": "RB",
    "S2": "advmod",
    "S1": ["dep", "parataxis"]
  },
  "although": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  },
  "and": {
    "POS": "CC",
    "S2": "cc",
    "S1": ["conj"]
  },
  "then": {
    "POS": "RB",
    "S2": "advmod",
    "S1": ["dep", "parataxis"] # previous sentence is always S1
  },
  "for example": {
    "POS": {"for": "IN", "example": "NN"},
    "S2": "nmod",
    "S1": ["dep", "parataxis"]
  },
  "before": {
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  },
  "meanwhile": {
    "POS": "RB",
    "S2": "advmod",
    "S1": ["parataxis", "dep"]
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
  "but": {
    "POS": "CC",
    "S2": "cc",
    "S1": ["conj"]
  },
  "because": {
    "POS": "IN",
    "S2": "mark", # S2 head (full S head) ---> connective
    "S1": ["advcl"], # S1 head ---> S2 head (full S head)
    # "alternates": ["just because", "only because"], # fix me (????)
    # "lost_alternates": ["because of"]
  },
  "as": {
    # e.g. "as only about 4 people showed up, 
    #       it was more of a discussion than a formal talk"
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
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
  "when": {
    # e.g. "when stopped by, he asked how you were"
    "POS": "WRB",
    "S2": "advmod",
    "S1": ["advcl"]
  },
  "while": {
    # e.g. "while he watched tv, he kept knitting the sweater"
    "POS": "IN",
    "S2": "mark",
    "S1": ["advcl"]
  }
}

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
use corenlp server (see https://github.com/erindb/corenlp-ec2-startup)
to parse sentences: tokens, dependency parse
"""
def get_parse(sentence):
  url = "http://localhost:12345?properties={annotators:'tokenize,ssplit,pos,depparse'}"
  data = sentence
  parse_string = requests.post(url, data=data).text
  return parse_string


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
  parse_data_directory("ptb")
  # parse_data_directory("wikitext-103")
  # t = parse_example("however", '{"sentences":[{"index":0,"basicDependencies":[{"dep":"ROOT","governor":0,"governorGloss":"ROOT","dependent":4,"dependentGloss":"however"},{"dep":"nsubj","governor":4,"governorGloss":"however","dependent":1,"dependentGloss":"It"},{"dep":"aux","governor":4,"governorGloss":"however","dependent":2,"dependentGloss":"did"},{"dep":"neg","governor":4,"governorGloss":"however","dependent":3,"dependentGloss":"not"},{"dep":"punct","governor":6,"governorGloss":"cover","dependent":5,"dependentGloss":","},{"dep":"dep","governor":4,"governorGloss":"however","dependent":6,"dependentGloss":"cover"},{"dep":"det","governor":8,"governorGloss":"sort","dependent":7,"dependentGloss":"any"},{"dep":"dobj","governor":6,"governorGloss":"cover","dependent":8,"dependentGloss":"sort"},{"dep":"case","governor":11,"governorGloss":"taxes","dependent":9,"dependentGloss":"of"},{"dep":"amod","governor":11,"governorGloss":"taxes","dependent":10,"dependentGloss":"local"},{"dep":"nmod","governor":8,"governorGloss":"sort","dependent":11,"dependentGloss":"taxes"},{"dep":"cc","governor":11,"governorGloss":"taxes","dependent":12,"dependentGloss":"or"},{"dep":"amod","governor":14,"governorGloss":"measures","dependent":13,"dependentGloss":"similar"},{"dep":"conj","governor":11,"governorGloss":"taxes","dependent":14,"dependentGloss":"measures"}],"enhancedDependencies":[{"dep":"ROOT","governor":0,"governorGloss":"ROOT","dependent":4,"dependentGloss":"however"},{"dep":"nsubj","governor":4,"governorGloss":"however","dependent":1,"dependentGloss":"It"},{"dep":"aux","governor":4,"governorGloss":"however","dependent":2,"dependentGloss":"did"},{"dep":"neg","governor":4,"governorGloss":"however","dependent":3,"dependentGloss":"not"},{"dep":"punct","governor":6,"governorGloss":"cover","dependent":5,"dependentGloss":","},{"dep":"dep","governor":4,"governorGloss":"however","dependent":6,"dependentGloss":"cover"},{"dep":"det","governor":8,"governorGloss":"sort","dependent":7,"dependentGloss":"any"},{"dep":"dobj","governor":6,"governorGloss":"cover","dependent":8,"dependentGloss":"sort"},{"dep":"case","governor":11,"governorGloss":"taxes","dependent":9,"dependentGloss":"of"},{"dep":"amod","governor":11,"governorGloss":"taxes","dependent":10,"dependentGloss":"local"},{"dep":"nmod:of","governor":8,"governorGloss":"sort","dependent":11,"dependentGloss":"taxes"},{"dep":"cc","governor":11,"governorGloss":"taxes","dependent":12,"dependentGloss":"or"},{"dep":"amod","governor":14,"governorGloss":"measures","dependent":13,"dependentGloss":"similar"},{"dep":"nmod:of","governor":8,"governorGloss":"sort","dependent":14,"dependentGloss":"measures"},{"dep":"conj:or","governor":11,"governorGloss":"taxes","dependent":14,"dependentGloss":"measures"}],"enhancedPlusPlusDependencies":[{"dep":"ROOT","governor":0,"governorGloss":"ROOT","dependent":4,"dependentGloss":"however"},{"dep":"nsubj","governor":4,"governorGloss":"however","dependent":1,"dependentGloss":"It"},{"dep":"aux","governor":4,"governorGloss":"however","dependent":2,"dependentGloss":"did"},{"dep":"neg","governor":4,"governorGloss":"however","dependent":3,"dependentGloss":"not"},{"dep":"punct","governor":6,"governorGloss":"cover","dependent":5,"dependentGloss":","},{"dep":"dep","governor":4,"governorGloss":"however","dependent":6,"dependentGloss":"cover"},{"dep":"det","governor":8,"governorGloss":"sort","dependent":7,"dependentGloss":"any"},{"dep":"dobj","governor":6,"governorGloss":"cover","dependent":8,"dependentGloss":"sort"},{"dep":"case","governor":11,"governorGloss":"taxes","dependent":9,"dependentGloss":"of"},{"dep":"amod","governor":11,"governorGloss":"taxes","dependent":10,"dependentGloss":"local"},{"dep":"nmod:of","governor":8,"governorGloss":"sort","dependent":11,"dependentGloss":"taxes"},{"dep":"cc","governor":11,"governorGloss":"taxes","dependent":12,"dependentGloss":"or"},{"dep":"amod","governor":14,"governorGloss":"measures","dependent":13,"dependentGloss":"similar"},{"dep":"nmod:of","governor":8,"governorGloss":"sort","dependent":14,"dependentGloss":"measures"},{"dep":"conj:or","governor":11,"governorGloss":"taxes","dependent":14,"dependentGloss":"measures"}],"tokens":[{"index":1,"word":"It","originalText":"It","characterOffsetBegin":0,"characterOffsetEnd":2,"pos":"PRP","before":"","after":" "},{"index":2,"word":"did","originalText":"did","characterOffsetBegin":3,"characterOffsetEnd":6,"pos":"VBD","before":" ","after":" "},{"index":3,"word":"not","originalText":"not","characterOffsetBegin":7,"characterOffsetEnd":10,"pos":"RB","before":" ","after":" "},{"index":4,"word":"however","originalText":"however","characterOffsetBegin":11,"characterOffsetEnd":18,"pos":"RB","before":" ","after":" "},{"index":5,"word":",","originalText":",","characterOffsetBegin":19,"characterOffsetEnd":20,"pos":",","before":" ","after":" "},{"index":6,"word":"cover","originalText":"cover","characterOffsetBegin":21,"characterOffsetEnd":26,"pos":"VB","before":" ","after":" "},{"index":7,"word":"any","originalText":"any","characterOffsetBegin":27,"characterOffsetEnd":30,"pos":"DT","before":" ","after":" "},{"index":8,"word":"sort","originalText":"sort","characterOffsetBegin":31,"characterOffsetEnd":35,"pos":"NN","before":" ","after":" "},{"index":9,"word":"of","originalText":"of","characterOffsetBegin":36,"characterOffsetEnd":38,"pos":"IN","before":" ","after":" "},{"index":10,"word":"local","originalText":"local","characterOffsetBegin":39,"characterOffsetEnd":44,"pos":"JJ","before":" ","after":" "},{"index":11,"word":"taxes","originalText":"taxes","characterOffsetBegin":45,"characterOffsetEnd":50,"pos":"NNS","before":" ","after":" "},{"index":12,"word":"or","originalText":"or","characterOffsetBegin":51,"characterOffsetEnd":53,"pos":"CC","before":" ","after":" "},{"index":13,"word":"similar","originalText":"similar","characterOffsetBegin":54,"characterOffsetEnd":61,"pos":"JJ","before":" ","after":" "},{"index":14,"word":"measures","originalText":"measures","characterOffsetBegin":62,"characterOffsetEnd":70,"pos":"NNS","before":" ","after":""}]}]}', "current", "prev")
  # print(t)
