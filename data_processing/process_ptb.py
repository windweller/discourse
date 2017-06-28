#! /usr/bin/env python

"""

Given the mrg files from LDC99t42, create text files for each discourse marker
with sentences (or pairs of sentences) that include that discourse marker.

e.g. wsj_BECAUSE.txt

At first, just news sentences. We can extend to more later.

"""

from nltk.corpus import LazyCorpusLoader
from nltk.corpus import CategorizedBracketParseCorpusReader
import re

training_sentences = {"because": [], "but": []}
discourse_markers = training_sentences.keys()

# initialize loader
ptb = LazyCorpusLoader( # Penn Treebank v3: WSJ and Brown portions
    'ptb', CategorizedBracketParseCorpusReader,
    r'(wsj/\d\d/wsj_\d\d|brown/c[a-z]/c[a-z])\d\d.mrg',
    cat_file='allcats.txt', tagset='wsj')

# exclude whatever the "-NONE-" tag is...
def get_words(s):
	return [pair[0] for pair in s if pair[1] != "-NONE-"]

print(str(len(ptb.fileids('news'))) + " documents")

num_sentences = 0
# restrict to news sentences:
for fileid in ptb.fileids('news'):
	sentences = ptb.tagged_sents(fileid)
	num_sentences += len(sentences)
	for s in sentences:
		words = get_words(s)
		for marker in discourse_markers:
			if marker in words:
				print " ".join(words)

print(str(num_sentences) + " sentences")
