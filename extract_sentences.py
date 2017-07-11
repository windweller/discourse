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

# instead, and then, even though*, although*, furthermore, 

discourse_markers = ["but", "because", "furthermore", "therefore"]


def extract_because():

	data_dir = "data/ptb/"
	parses_dir = os.path.join(data_dir, "parses/")
	if not os.path.isdir(parses_dir):
	  os.mkdir(parses_dir)

	parse_string = '{"sentences":[{"index":0,"basicDependencies":[{"dep":"ROOT","governor":0,"governorGloss":"ROOT","dependent":2,"dependentGloss":"like"},{"dep":"nsubj","governor":2,"governorGloss":"like","dependent":1,"dependentGloss":"I"},{"dep":"dobj","governor":2,"governorGloss":"like","dependent":3,"dependentGloss":"her"},{"dep":"mark","governor":7,"governorGloss":"nice","dependent":4,"dependentGloss":"because"},{"dep":"nsubj","governor":7,"governorGloss":"nice","dependent":5,"dependentGloss":"she"},{"dep":"cop","governor":7,"governorGloss":"nice","dependent":6,"dependentGloss":"is"},{"dep":"advcl","governor":2,"governorGloss":"like","dependent":7,"dependentGloss":"nice"}],"enhancedDependencies":[{"dep":"ROOT","governor":0,"governorGloss":"ROOT","dependent":2,"dependentGloss":"like"},{"dep":"nsubj","governor":2,"governorGloss":"like","dependent":1,"dependentGloss":"I"},{"dep":"dobj","governor":2,"governorGloss":"like","dependent":3,"dependentGloss":"her"},{"dep":"mark","governor":7,"governorGloss":"nice","dependent":4,"dependentGloss":"because"},{"dep":"nsubj","governor":7,"governorGloss":"nice","dependent":5,"dependentGloss":"she"},{"dep":"cop","governor":7,"governorGloss":"nice","dependent":6,"dependentGloss":"is"},{"dep":"advcl:because","governor":2,"governorGloss":"like","dependent":7,"dependentGloss":"nice"}],"enhancedPlusPlusDependencies":[{"dep":"ROOT","governor":0,"governorGloss":"ROOT","dependent":2,"dependentGloss":"like"},{"dep":"nsubj","governor":2,"governorGloss":"like","dependent":1,"dependentGloss":"I"},{"dep":"dobj","governor":2,"governorGloss":"like","dependent":3,"dependentGloss":"her"},{"dep":"mark","governor":7,"governorGloss":"nice","dependent":4,"dependentGloss":"because"},{"dep":"nsubj","governor":7,"governorGloss":"nice","dependent":5,"dependentGloss":"she"},{"dep":"cop","governor":7,"governorGloss":"nice","dependent":6,"dependentGloss":"is"},{"dep":"advcl:because","governor":2,"governorGloss":"like","dependent":7,"dependentGloss":"nice"}],"tokens":[{"index":1,"word":"I","originalText":"I","characterOffsetBegin":0,"characterOffsetEnd":1,"pos":"PRP","before":"","after":" "},{"index":2,"word":"like","originalText":"like","characterOffsetBegin":2,"characterOffsetEnd":6,"pos":"VBP","before":" ","after":" "},{"index":3,"word":"her","originalText":"her","characterOffsetBegin":7,"characterOffsetEnd":10,"pos":"PRP$","before":" ","after":" "},{"index":4,"word":"because","originalText":"because","characterOffsetBegin":11,"characterOffsetEnd":18,"pos":"IN","before":" ","after":" "},{"index":5,"word":"she","originalText":"she","characterOffsetBegin":19,"characterOffsetEnd":22,"pos":"PRP","before":" ","after":" "},{"index":6,"word":"is","originalText":"is","characterOffsetBegin":23,"characterOffsetEnd":25,"pos":"VBZ","before":" ","after":" "},{"index":7,"word":"nice","originalText":"nice","characterOffsetBegin":26,"characterOffsetEnd":30,"pos":"JJ","before":" ","after":""}]}]}'

	parse = json.loads(parse_string)

	because_dependency_types = {
	  "cause": "mark",
	  "effect": "advcl"
	}

	class Sentence():
	  def __init__(self, json_sentence):
	    self.json = json_sentence
	    self.dependencies = json_sentence["basicDependencies"]
	    self.tokens = json_sentence["tokens"]
	  def pairs(self, discourse_marker):
	    None
	  def index(self, word):
	    return [t["word"] for t in self.tokens].index(word) + 1
	  def token(self, index):
	    return self.tokens[index-1]
	  def word(self, index):
	    return self.token(index)["word"]

	sentence = Sentence(parse["sentences"][0])
	# sentence.pairs("because")

	print(sentence.index("because"))


def indices(lst, element):
  # https://stackoverflow.com/a/18669080
  result = []
  offset = -1
  while True:
    try:
      offset = lst.index(element, offset+1)
    except ValueError:
      return result
    result.append(offset)


def make_filename(split, marker, s, data_dir):
  return os.path.join(data_dir,
         split + "_" + marker.upper() + "_S" + str(s+1) + ".txt")


def text_filenames(data_dir):
  return [os.path.join(data_dir, split + "_" + label + ".txt") \
                      for label in ["S1", "S2", "labels"] \
                      for split in ["train", "valid", "test"]]


def get_examples_from_words(marker, words, line, previous_line):
  example_list = []
  # fix me to deal with multiword discourse markers
  for i in indices(words, marker):
    if marker=="because" and i==0:
      # fix me to use dependency parse
      None
    elif i==0:
      example_list.append((previous_line, line, marker))
    else:
      before = " ".join(words[:i])
      after = " ".join(words[i+1:])
      example_list.append((before, after, marker))
  return example_list


def write_example_file(data_dir, split, label, element_list):
  savepath = os.path.join(data_dir, split + "_" + label + ".txt")
  w = open(savepath, "w")
  w.write("\n".join(element_list))
  w.close()


def ptb():
  ## PTB
  # Given a corpus where each line is a sentence, find examples with given
  # discourse markers.

  num_sentences = 0

  # download ptb if it doesn't exist
  # fix me
  if not os.path.isfile("data/ptb/ptb.test.txt"):
    # download it
    None
  if not os.path.isfile("data/ptb/ptb.train.txt"):
    None
  if not os.path.isfile("data/ptb/ptb.valid.txt"):
    None

  # check if text files already exist
  if all([os.path.isfile(f) for f in text_filenames("data/ptb/")]):
    print("PTB text files already exist. " +
          "Delete them if you want to rerun.")
  else:
    for split in ["train", "valid", "test"]:

      examples = []

      datapath = "data/ptb/ptb." + split + ".txt"
      fd = open(datapath)
      line = fd.readline().strip()
      previous_line = ""
      while line:
        words = line.split()
        for marker in discourse_markers:
          example_list = get_examples_from_words(marker, words, line, previous_line)
          examples += example_list

        previous_line = line
        line = fd.readline().strip()
        num_sentences += 1
      fd.close()

      s1, s2, labels = zip(*examples)

      write_example_file("data/ptb/", split, "S1", s1)
      write_example_file("data/ptb/", split, "S2", s2)
      write_example_file("data/ptb/", split, "labels", labels)


## Wikitext 103

def wikitext(COLLAPSE_NUMBERS=False, STRIP_PUNCTUATION=False):

  def collapse_numbers(s):
    if COLLAPSE_NUMBERS:
      return re.sub("([^ ]*)\d([^ ]*)", "\1N\2", s)
    else:
      return s

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

extract_because()

# dependencies = 
# print([d for d in dependencies if d['dependentGloss']=="because"])
# print([d for d in dependencies if d['governor']==4])

# for line in corpus:
  # regex search for discourse marker
    # if found, parse
    # for each dependency from discourse marker
      # find head; its phrase = S1
      # find one dependency from head: its phrase = S2
# ptb()
# wikitext()
# winograd()