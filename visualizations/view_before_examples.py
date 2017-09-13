import nltk

extract_num = 20
previous_line = "" 
n = 0


import sys
reload(sys)
sys.setdefaultencoding('latin-1')


class bcolors:
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

s1_col = bcolors.BLUE
s2_col = bcolors.BLUE
before_col = bcolors.BOLD + bcolors.GREEN 


tokens = open("data/wikitext-103/wiki.valid.tokens").read().replace("\n", ". ")
print("tokenizing...")
sent_list = nltk.sent_tokenize(tokens)
print("tokenization complete. total number of sentences: " + str(len(sent_list)))
 
marker = "also"

for sent in sent_list:
	if marker in sent:
		n+=1
		sent_to_print = prev_sent + "\n" + s1_col
		sent_to_print += sent.replace(marker, bcolors.ENDC + before_col + marker + bcolors.ENDC + s2_col)
		sent_to_print += bcolors.ENDC
		print ("actual " + str(n) + ".")
		print (sent_to_print)
		print ("")
	if n==extract_num:
		break
	prev_sent = sent

import pickle
import tensorflow as tf

def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

stop

data = pickle.load(open("data/wikitext-103/valid_all.ids.pkl", "rb"))
class_labels = pickle.load(open("data/wikitext-103/class_labels_all.pkl", "rb"))

vocab, rev_vocab = initialize_vocab("data/wikitext-103/vocab.dat")
print(rev_vocab[0:10])

class_lst = [c for c in class_labels]
for c in class_labels:
	class_lst[class_labels[c]] = c

before = [d for d in data if d[2] == class_labels[marker]][0:20]

n=0
for s1, s2, label in before:
	n+=1
	sent_to_print = s1_col + " ".join([rev_vocab[w] for w in s1])
	sent_to_print += before_col + " " + marker + " "
	sent_to_print += s2_col + " ".join([rev_vocab[w] for w in s2])
	sent_to_print += bcolors.ENDC
	print ("ours " + str(n) + ".")
	print (sent_to_print)
	print ("")


