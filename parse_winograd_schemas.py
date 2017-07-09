#! /usr/bin/bash python

"""
parse this file:
http://www.cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WSCollection.xml
into two files containing the two disambiguated sentences

"""

data_dir = "data"

import xml.etree.ElementTree as ET
import re
import os

tree = ET.parse('data/winograd/WSCollection.xml')
root = tree.getroot()

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

	if "because" in original_sentence.split():

		answers_elements = schema.find('answers').findall("answer")
		answers = [e.text.strip() for e in answers_elements]

		for version in ["incorrect", "correct"]:
			if version == "correct":
				index = correct_index
			else:
				index = 1-correct_index

			noun_phrase = answers[index]
			sentence = " ".join([start, noun_phrase, end])
			sentences[version].append(sentence)

			all_sentences.append(sentence)

# print(sentences["correct"])
# print(len(sentences["correct"]))

# for version in ["correct", "incorrect"]:
# 	filename = "winograd_" + version.upper() + ".txt"
# 	filepath = os.path.join(data_dir, "winograd", filename)
# 	open(filepath, "w").write("\n".join(sentences[version]))

filepath = os.path.join(data_dir, "winograd", "valid_BECAUSE.txt")
open(filepath, "w").write("\n".join(all_sentences))

