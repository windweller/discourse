#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
NOTE: if "because" starts a sentence, it's ambiguous and we shouldn't keep it.
E.g.:
"I don't know. Because there are good reasons to do it one way, but ..."
"Because there are so many options, most people are unsure what ..."
"""

import re
import os
import json
import pickle
import requests

def collapse_numbers(s):
  if COLLAPSE_NUMBERS:
    return re.sub("([^ ]*)\d([^ ]*)", "\1N\2", s)
  else:
    return s

# instead, and then, even though*, although*, furthermore, 
# candidate list on slack, vote on them
# grab the most frequent of those from papers we read.

# regex for words
# depparse
# search for dep patterns we're looking for

# list of patterns that we're looking for
# log which ones we ignore
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
    "POS": "RB",
    "S2": "advmod",
    "S1": ["dep", "parataxis"]
  },
  "still": {
    "POS": "RB",
    "S2": "advmod",
    "S1": ["parataxis", "dep"]
  },
  "though": {
    "POS": "RB",
    "S2": "advmod",
    "S1": ["parataxis", "dep"]
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
    "alternates": ["just because", "only because"], # fix me (????)
    "lost_alternates": ["because of"]
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

def get_indices(lst, element):
  # https://stackoverflow.com/a/18669080
  result = []
  offset = -1
  while True:
    try:
      offset = lst.index(element, offset+1)
    except ValueError:
      return result
    result.append(offset)

# for each dicourse word, if it's the part of speech we expect and it's
# hanging off the head word

discourse_markers = dependency_types.keys()

cached_parses = {}
if os.path.isfile("cached_parses.dat"):
  cached_parses = pickle.load(open("cached_parses.dat", "rb"))

def get_parse(sentence):
  sentence_tag = re.sub("[^A-z]", "", sentence)
  if sentence_tag in cached_parses:
    parse_string = cached_parses[sentence_tag]
  else:
    url = "http://localhost:12345?properties={annotators:'tokenize,ssplit,pos,depparse'}"
    data = sentence
    parse_string = requests.post(url, data=data).text
    cached_parses[sentence_tag] = parse_string
  return parse_string

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

  marker_index = sentence.indices(marker)[0] # fix me!!!
  dep_patterns = dependency_types[marker]

  possible_marker_indices = sentence.indices(marker)
  possible_s2_head_indices = [p for marker_index in possible_marker_indices for p in sentence.find_parents(
    marker_index,
    filter_types=[dep_patterns["S2"]]
  )]

  # Look for S2
  possible_s2_head_indices = sentence.find_parents(
    marker_index,
    filter_types=[dep_patterns["S2"]]
  )
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
    print("NO MATCH: " + current_sentence)
    return None



def make_filename(split, marker, s, data_dir):
  return os.path.join(data_dir,
         split + "_" + marker.upper() + "_S" + str(s+1) + ".txt")


def text_filenames(data_dir):
  return [os.path.join(data_dir, split + "_" + label + ".txt") \
                      for label in ["S1", "S2", "labels"] \
                      for split in ["train", "valid", "test"]]



def write_example_file(data_dir, split, label, element_list):
  savepath = os.path.join(data_dir, split + "_" + label + ".txt")
  w = open(savepath, "w")
  w.write("\n".join(element_list))
  w.close()


def parse_data_directory(data_tag, tag2, extension, collapse_nums=False, strip_punctuation=False):
  ## PTB
  # Given a corpus where each line is a sentence, find examples with given
  # discourse markers.

  data_dir = "data/" + data_tag + "/"

  num_sentences = 0

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
  if all([os.path.isfile(f) for f in text_filenames(data_dir)]):
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
        if len(line) > 0:
          if data_dir=="data/wikitext-103" and line_words[0] == "=":
            previous_line = ""
          else:
            if data_dir=="data/wikitext-103":
              doc_sentences = line.split(" . ")
            else:
              doc_sentences = [line]
            for s in doc_sentences:
              if collapse_nums:
                words = collapse_numbers(s)

              if strip_punctuation:
                words_to_exclude = ["@-@", ".", "\"", ",", ":",
                                    "—", "(", ")", "@,@", "@.@",
                                    ";", "'", "–", "!", "?"]
                s = " ".join([w for w in s.split() if not w in words_to_exclude])

              pattern = " (" + "|".join(discourse_markers) + ") "
              m = re.match(pattern, s)
              if m:
                parse_string = get_parse(s)
                for marker in m.groups():
                  example_tuple = parse_example(marker, parse_string, s, previous_line)
                  if example_tuple:
                    s1, s2, label = example_tuple
                    s1s.append(s1.strip())
                    s2s.append(s2.strip())
                    labels.append(label.strip())
              previous_line = s

        previous_line = line
        line = fd.readline()
        num_sentences += 1
      fd.close()

      write_example_file(data_dir, split, "S1", s1s)
      write_example_file(data_dir, split, "S2", s2s)
      write_example_file(data_dir, split, "labels", labels)


## Wikitext 103

def wikitext():

  # download wikitext if it doesn't exist
  # wiki.test.tokens
  # wiki.train.tokens
  # wiki.valid.tokens

  # check if text files already exist
  if all([os.path.isfile(f) for f in text_filenames("data/wikitext-103/")]):
    print("Wikitext-103 text files already exist. " +
          "Delete them if you want to rerun.")
  else:
    num_sentences = 0

    for split in ["train", "valid", "test"]:

      examples = []

      datapath = "data/wikitext-103/wiki." + split + ".tokens"
      fd = open(datapath)
      line = fd.readline()
      previous_line = ""

      while line:
        line_words = line.split()
        if len(line_words) > 0:
          if line_words[0] == "=":
            previous_line = ""
          else:
            doc_sentences = " ".join(line_words).split(" . ")
            for s in doc_sentences:
              words = collapse_numbers(s).split()

              if STRIP_PUNCTUATION:
                words_to_exclude = ["@-@", ".", "\"", ",", ":",
                                    "—", "(", ")", "@,@", "@.@",
                                    ";", "'", "–", "!", "?"]
                words = [w for w in words if not w in words_to_exclude]
              
              for marker in discourse_markers:
                example_list = get_examples_from_words(marker, words, line, previous_line)
                examples += example_list

        previous_line = line
        line = fd.readline()
        num_sentences += 1
      fd.close()

      s1, s2, labels = zip(*examples)

      write_example_file("data/wikitext-103/", split, "S1", s1)
      write_example_file("data/wikitext-103/", split, "S2", s2)
      write_example_file("data/wikitext-103/", split, "labels", labels)


## Winograd
# http://www.cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WSCollection.xml
def winograd():
  # download winograd if it doesn't exist

  # check if text files already exist
  if all([os.path.isfile(f) for f in ["data/winograd/valid_BECAUSE_S1.txt",
                                      "data/winograd/valid_BECAUSE_S2.txt"]]):
    print("Winograd text files already exist. " +
          "Delete them if you want to rerun.")
  else:

    import xml.etree.ElementTree as ET
    import os

    tree = ET.parse('data/winograd/WSCollection.xml')
    root = tree.getroot()

    def tokenize_like_wikitext(s):
      patterns = [("\.", " ."), (";", " ;"), ("'", " '"), (",", " ,"),
                  (" \"", " \" "), ("\" ", " \" "), ("\!", " !")]
      for pattern, replacement in patterns:
        s = re.sub(pattern, replacement, s)
      return s.split()

    n_pairs = 0

    sentences = {"correct": [], "incorrect": []}
    all_sentences = []

    # for each schema pair of sentences
    for schema in root:

      # get correct answer
      correct_answer_text = schema.find('correctAnswer').text
      correct = re.sub("\.", "", correct_answer_text.strip())
      assert (correct=="A" or correct=="B")
      correct_index = ["A", "B"].index(correct)

      pronoun = schema.find('quote').find("pron").text.strip()
      start = schema.find('text').find("txt1").text.strip()
      end = schema.find('text').find("txt2").text.strip()

      start = re.sub("\n", " ", start)
      end = re.sub("\n", " ", end)
      start = re.sub("  ", " ", start)
      end = re.sub("  ", " ", end)

      original_sentence = " ".join([start, pronoun, end])

      words = tokenize_like_wikitext(original_sentence)
      # print(" ".join(words))
      if "because" in words:
        because_index = words.index("because")

        answers_elements = schema.find('answers').findall("answer")
        answers = [e.text.strip() for e in answers_elements]

        for version in ["incorrect", "correct"]:
          if version == "correct":
            index = correct_index
          else:
            index = 1-correct_index

          noun_phrase = answers[index]
          sentence = " ".join([start, noun_phrase, end])

          before = " ".join(words[:because_index])
          after = " ".join(words[because_index+1:])
          sentence_pair = (before, after)
          all_sentences.append(sentence_pair)

    for s in [0, 1]:
      filepath = "data/winograd/valid_BECAUSE_S" + str(s+1) + ".txt"
      open(filepath, "w").write("\n".join([pair[s] for pair in all_sentences]))


parse_data_directory("ptb", "ptb", "txt")
parse_data_directory("wikitext-103", "wiki", "tokens")
pickle.dump(cached_parses, open("cached_parses.dat", "wb"))

# for line in corpus:
  # regex search for discourse marker
    # if found, parse
    # for each dependency from discourse marker
      # find head; its phrase = S1
      # find one dependency from head: its phrase = S2
# ptb()
# wikitext()
# winograd()