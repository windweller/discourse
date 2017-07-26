from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange
import tensorflow as tf

import pickle
import os
import sys
import logging
from os.path import join as pjoin

import data

np.random.seed(123)

FLAGS = tf.app.flags.FLAGS

## fix me
def tokenize(string):
    return [int(s) for s in string.split()]

def get_label(marker):
    # now in data_dir/"class_labels.pkl"
    assert marker in ["because", "but"]
    if marker=="because":
        return 0
    else:
        return 1

"""
fnamex: e.g. train_BECAUSE.ids.txt

x: sentence (1st chunk, before discourse marker)
x2: sentence (2nd chunk, after discourse marker)
y: classification (label 0: because ; 1:but)

cause and effect: load in one file

2 pair_iter, one for each task
but wrap both in one if i have time with a flag
"""

def get_training_tuple(tokens1, tokens2, task, marker, vocab,
                       rev_vocab, winograd_label=None):

    assert(marker=="because" or marker=="but")

    previous_sentence_tokens = tokens1
    next_sentence_tokens = tokens2

    tokens = tokens1 + [vocab[marker]] + tokens2

    if marker=="because":
        # exclude "of" from next_sentence for because
        if next_sentence_tokens[0] == vocab["of"]:
            next_sentence_tokens = next_sentence_tokens[1:]

    # exclude sentences that are too long
    if len(previous_sentence_tokens) > FLAGS.max_seq_len \
            or len(next_sentence_tokens) > FLAGS.max_seq_len:
        return None

    training_example = []

    if task=="cause_effect":
        effect_tokens = previous_sentence_tokens
        cause_tokens = next_sentence_tokens
        if np.random.randint(0, 2):
            training_example += [effect_tokens, cause_tokens, 0]
        else:
            training_example += [cause_tokens, effect_tokens, 1]
    else:
        training_example += [previous_sentence_tokens, next_sentence_tokens,
                             get_label(marker)]

    if task=="winograd":
        # winograd task gets winograd labels
        training_example.append(winograd_label)

    text = " ".join([rev_vocab[t] for t in tokens])
    training_example.append(text)

    return tuple(training_example)

def build_batches(task, data_dir, split, vocab, rev_vocab, batch_size,
                  shuffle, cache):

    # build dictionary of files to look at, based on task
    fds = {}
    fname_because_s1 = pjoin(data_dir, split + "_BECAUSE_S1.ids.txt")
    fname_but_s1 = pjoin(data_dir, split + "_BUT_S1.ids.txt")
    fname_because_s2 = pjoin(data_dir, split + "_BECAUSE_S2.ids.txt")
    fname_but_s2 = pjoin(data_dir, split + "_BUT_S2.ids.txt")

    discourse_markers = ["but", "because"]

    fds["because1"] = open(fname_because_s1)
    fds["because2"] = open(fname_because_s2)
    if task=="but_because":
        fds["but1"] = open(fname_but_s1)
        fds["but2"] = open(fname_but_s2)

    # read lines from relevant files
    lines = {filetag: fds[filetag].readline() for filetag in fds.keys()}

    all_training_tuples = []
    if task=="winograd":
        winograd_label = 0
    else:
        winograd_label = None

    while all(lines.values()):
        matched_tuples = []

        for marker in discourse_markers:
            tokens1 = tokenize(lines[marker + "1"])
            tokens2 = tokenize(lines[marker + "2"])
            
            training_tuple = get_training_tuple(tokens1=tokens1,
                                                tokens2=tokens2, task=task,
                                                marker=marker, vocab=vocab,
                                                rev_vocab=rev_vocab,
                                                winograd_label=winograd_label)

            if training_tuple: matched_tuples.append(training_tuple)

        # get rid of both training examples, if we're trying to match
        # (e.g.) the number of but and because sentences
        if all(matched_tuples): all_training_tuples += matched_tuples

        if task=="winograd":
            winograd_label = 1-winograd_label

        lines = {filetag: fds[filetag].readline() for filetag in fds.keys()}

    if shuffle:
        np.random.shuffle(all_training_tuples)

    batches = []

    for batch_start in xrange(0, len(all_training_tuples), batch_size):
        batch_end = batch_start + batch_size
        batch = zip(*all_training_tuples[batch_start:batch_end])
        batches.append(batch)

    if shuffle:
        np.random.shuffle(batches)

    if cache:
        cache_filename = pjoin(data_dir, split + "_" + str(batch_size) + ".pkl")
        pickle.dump(batches, open(cache_filename, "wb"))

    return batches

def pair_iter(task, data_dir, split, vocab, rev_vocab, batch_size,
              shuffle=True, cache=False):
    """Iterator to get batches of inputs for models

    Keyword arguments:
    data_dir -- data directory
    split -- train, valid, or test
    vocab -- a dict from words to their ids in vocab
    rev_vocab -- a dict from ids to words
    batch_size -- number of sentences per batch
    shuffle -- we don't want to shuffle the validation and test sets.
    cache -- sometimes we want to cache files so that random shuffle is the same
    """

    assert(task in ["winograd", "but_because", "cause_effect"])
    assert(split in ["train", "valid", "test"])

    if cache:
        cache_filename = pjoin(data_dir, split + "_" + str(batch_size) + ".pkl")
        if os.path.isfile(cache_filename):
            logging.info("restoring old batches")
            batches = pickle.load(open(cache_filename, 'rb'))
    if not cache or not (os.path.isfile(cache_filename)):
        logging.info("generate new batches")
        batches = build_batches(task=task, data_dir=data_dir, split=split,
                                vocab=vocab, rev_vocab=rev_vocab,
                                batch_size = batch_size,
                                shuffle=shuffle, cache=cache)
    while True:
        if len(batches) == 0:
            # stopping condition, when batches is empty
            break

        if task=="winograd":
            s1_tokens, s2_tokens, y, winograd_label, text = batches.pop(0)
        else:
            s1_tokens, s2_tokens, y, text = batches.pop(0)

        # pad sentence chunks
        s1_padded, s2_padded = padded(s1_tokens), padded(s2_tokens)

        # first sentence (e.g. before discourse marker)
        s1_tokens = np.array(s1_padded)
        s1_mask = (s1_tokens != data.PAD_ID).astype(np.int32)

        # second sentence (e.g. after discourse marker)
        s2_tokens = np.array(s2_padded)
        s2_mask = (s2_tokens != data.PAD_ID).astype(np.int32)

        if task=="winograd":
            yield (s1_tokens, s1_mask, s2_tokens, s2_mask, y, winograd_label, text)
        else:
            yield (s1_tokens, s1_mask, s2_tokens, s2_mask, y, text)

    return


def padded(tokens, batch_pad=0):
    maxlen = max(map(lambda x: len(x), tokens)) if batch_pad == 0 else batch_pad
    return map(lambda token_list: token_list + [data.PAD_ID] * (maxlen - len(token_list)), tokens)


# data_dir, split, vocab, batch_size
if __name__ == '__main__':

    tf.flags.DEFINE_integer("max_seq_len", 35, "cut off sentence after this of words")

    from tensorflow.python.platform import gfile
    def initialize_vocabulary(vocabulary_path):
        # map vocab to word embeddings
        if gfile.Exists(vocabulary_path):
            rev_vocab = []
            with gfile.GFile(vocabulary_path, mode="r") as f:
                rev_vocab.extend(f.readlines())
            rev_vocab = [line.strip('\n') for line in rev_vocab]
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
            return vocab, rev_vocab
        else:
            raise ValueError("Vocabulary file %s not found.", vocabulary_path)
    vocab, rev_vocab = initialize_vocabulary(pjoin("data/ptb", "vocab.dat"))

    # print(next(winograd_pair_iter("data/ptb/", vocab=vocab, batch_size=10, shuffle=False)))
    output = next(pair_iter(task="but_because", data_dir="data/ptb",
                            split="valid", vocab=vocab, rev_vocab=rev_vocab,
                            batch_size=10, shuffle=True, cache=False))
    print(output)
