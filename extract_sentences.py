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

discourse_markers = ["but", "because"]

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
  return [make_filename(split, marker, s, data_dir) \
                      for s in [0,1] \
                      for marker in ["because", "but"] \
                      for split in ["train", "valid", "test"]]

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
  print("PTB but/because text files already exist. " +
        "Delete them if you want to rerun.")
else:
  for split in ["train", "valid", "test"]:

    sentences = {m: [] for m in discourse_markers}

    datapath = "data/ptb/ptb." + split + ".txt"
    fd = open(datapath)
    line = fd.readline().strip()
    previous_line = ""
    while line:
      words = line.split()
      for marker in discourse_markers:
        for i in indices(words, marker):
          if marker!="because" and i==0:
            sentences[marker].append((previous_line, line))
          else:
            before = " ".join(words[:i])
            after = " ".join(words[i+1:])
            sentences[marker].append((before, after))

      previous_line = line
      line = fd.readline().strip()
      num_sentences += 1
    fd.close()

    for marker in discourse_markers:
      for s in [0, 1]:
        savepath = make_filename(split, marker, s, "data/ptb/")
        w = open(savepath, "w")
        w.write("\n".join([pair[s] for pair in sentences[marker]]))
        w.close()

## Wikitext 103

COLLAPSE_NUMBERS = False
STRIP_PUNCTUATION = False


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
  print("Wikitext-103 but/because text files already exist. " +
        "Delete them if you want to rerun.")
else:
  num_sentences = 0

  for split in ["train", "valid", "test"]:

    sentences = {m: [] for m in discourse_markers}

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
              for i in indices(words, marker):
                if marker!="because" and i==0:
                  sentences[marker].append((previous_line, line))
                else:
                  before = " ".join(words[:i])
                  after = " ".join(words[i+1:])
                  sentences[marker].append((before, after))

      previous_line = line
      line = fd.readline()
      num_sentences += 1
    fd.close()

    for marker in discourse_markers:
      for s in [0, 1]:
        savepath = make_filename(split, marker, s, "data/wikitext-103/")
        w = open(savepath, "w")
        w.write("\n".join([pair[s] for pair in sentences[marker]]))
        w.close()


## Winograd
# http://www.cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WSCollection.xml

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

