#! /usr/bin/env python

"""

Given the mrg files from LDC99t42, create text files for each discourse marker
with sentences (or pairs of sentences) that include that discourse marker.

e.g. wsj_BECAUSE.txt

At first, just WSJ sentences. We can extend to more later.

"""

from nltk.corpus import LazyCorpusLoader
from nltk.corpus import CategorizedBracketParseCorpusReader
import re

training_sentences = {"because": [], "but": []}
discourse_markers = training_sentences.keys()

ptb = LazyCorpusLoader( # Penn Treebank v3: WSJ and Brown portions
    'ptb', CategorizedBracketParseCorpusReader,
    r'(wsj/\d\d/wsj_\d\d|brown/c[a-z]/c[a-z])\d\d.mrg',
    cat_file='allcats.txt', tagset='wsj')

def get_words(s):
	return [pair[0] for pair in s if pair[1] != "-NONE-"]
	# return [w for w in words if not re.match("\*.*", w)]

print(str(len(ptb.fileids('news'))) + " documents")

num_sentences = 0
for fileid in ptb.fileids('news'):
	sentences = ptb.tagged_sents(fileid)
	num_sentences += len(sentences)
	for s in sentences:
		words = get_words(s)
		print(" ".join(words))

print(str(num_sentences) + " sentences")
	


# print(ptb.words('mrg/wsj/00/wsj_0010.mrg'))

# from nltk.corpus.reader.tagged import TaggedCorpusReader
# reader = TaggedCorpusReader("../data/ptb/mrg/wsj/00/", 'wsj_001.*.mrg')
# sentences = reader.sents()[1:]

# for s in sentences:
# 	print " ".join(s)

# 	>>> from nltk.corpus import ptb
# >>> print(ptb.fileids()) # doctest: +SKIP
# ['BROWN/CF/CF01.MRG', 'BROWN/CF/CF02.MRG', 'BROWN/CF/CF03.MRG', 'BROWN/CF/CF04.MRG', ...]
# >>> print(ptb.words('WSJ/00/WSJ_0003.MRG')) # doctest: +SKIP
# ['A', 'form', 'of', 'asbestos', 'once', 'used', '*', ...]
# >>> print(ptb.tagged_words('WSJ/00/WSJ_0003.MRG')) # doctest: +SKIP
# [('A', 'DT'), ('form', 'NN'), ('of', 'IN'), ...]

# for each directory in pos_dir

# for each file in that directory

# collect sentences for each discourse marker

