#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
from os.path import join as pjoin
from tensorflow.python.platform import gfile

from collections import Counter

import sys
reload(sys)
sys.setdefaultencoding('utf8')

_PAD = b"<pad>" # no need to pad
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1

max_ratio = 5.00
min_seq_len = 5
max_seq_len = 50

np.random.seed(123)


def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def load_data(rev_vocab, rev_labels):
	max_pretty_len = 15
	new_data = []
	markers = ["but", "because", "if", "when", "so"]

	print("(books)")
	old_data_test = pickle.load(open("data/books/test_but_because_when_if_for_example_so_before_still.ids.pkl", "rb"))
	# old_data_val = pickle.load(open("data/books/valid_but_because_when_if_for_example_so_before_still.ids.pkl", "rb"))

	# get for example from wikitext
	print("(wiki)")
	# wiki_data = pickle.load(open("data/wikitext-103/dep_all_sentence_pairs.pkl", "rb"))
	wiki_data = pickle.load(open("data/wikitext-103/wikitext-103_all_sentence_pairs.pkl", "rb"))

	max_pairs = 5000
	for marker in markers:
	# for s1, s2 in wiki_data["for example"]:
		n_pairs = 0
		n_pretty_examples = 0
		for s1, s2 in wiki_data[marker]:
			if float(len(s1))/len(s2) <= max_ratio and \
						float(len(s2))/len(s1) <= max_ratio and \
						len(s1)<max_seq_len and len(s2)<max_seq_len and \
						len(s1)>min_seq_len and len(s2)>min_seq_len:
				n_pairs += 1
				# if n_pretty_examples < 15:
				# 	if len(s1) < max_pretty_len and len(s2) < max_pretty_len:
				# 		n_pretty_examples += 1
				# 		print(" ".join(s1))
				# 		print("&" + marker + "&" )
				# 		print(" ".join(s2))
				# 		print("***********")
				if n_pairs > max_pairs:
					break
				# marker = "for example"
				words1 = " ".join(s1)
				words2 = " ".join(s2)
				new_data.append((words1, words2, marker))
		# print("***********")
		# print("***********")
		# print("***********")
		# print("***********")

	max_pairs = {marker: 8000 for marker in markers}
	n_pairs = {marker: 0 for marker in markers}
	old_data = old_data_test #+ old_data_val
	n_pretty_examples = 0
	for s1, s2, label in old_data:
		marker = rev_labels[label]
		if marker in markers:
			if n_pairs[marker] < max_pairs[marker]:
				# if marker != "for example":
				words1 = " ".join([rev_vocab[i] for i in s1])
				words2 = " ".join([rev_vocab[i] for i in s2])
				new_data.append((words1, words2, marker))
				n_pairs[marker] += 1
				# if n_pretty_examples < 15:
				# 	if len(s1) < max_pretty_len and len(s2) < max_pretty_len:
				# 		n_pretty_examples += 1
				# 		print(words1)
				# 		print("&" + marker + "&" )
				# 		print(words2)
				# 		print("***********")

	return new_data


if __name__ == '__main__':

	dataset = "wiki_and_books"

	if dataset=="books" or dataset =="wiki_and_books":
		class_labels = pickle.load(open("data/books/class_labels_but_because_when_if_for_example_so_before_still.pkl", "rb"))
		rev_labels = [marker for marker in class_labels]
		for marker in class_labels:
		    label = class_labels[marker]
		    rev_labels[label] = marker

		print("vocab")
		vocab, rev_vocab = initialize_vocabulary( "data/books/vocab_but_because_when_if_for_example_so_before_still.dat")
	else:
		rev_vocab = None
		rev_labels = None

	print("load data")
	full_dataset = load_data(rev_vocab, rev_labels)

	train_proportion = 0.9

	n_pairs = len(full_dataset)
	n_test = round((1-train_proportion)*n_pairs / 2)
	n_dev = n_test
	n_train = n_pairs - (n_test + n_dev)

	elements = ["s1", "s2", "labels"]
	splits = ["train", "dev", "test"]

	new_data = {split: {element: [] for element in elements} for split in splits}

	np.random.shuffle(full_dataset)

	print("split")
	for i in range(len(full_dataset)):

		s1, s2, label = full_dataset[i]

		if i < n_train:
			split = "train"
		elif i < n_train + n_dev:
			split = "dev"
		else:
			split = "test"

		new_data[split]["s1"].append(s1)
		new_data[split]["s2"].append(s2)
		new_data[split]["labels"].append(label)

	for split in splits:
		print(split)
		for element in elements:
			print(element)
			open("data/" + dataset + "/discourse_task_v1/{}.{}".format(element, split), "w").write("\n".join(new_data[split][element]))

	for split in splits:
		labels = new_data[split]["labels"]
		freqs = Counter(labels)
		print("*****")
		print(split)
		print(freqs)


