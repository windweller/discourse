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


# tf.flags.DEFINE_integer("max_seq_len", 35, "cut off sentence after this of words")

def tokenize(string):
    return [int(s) for s in string.split()]


"""
fnamex: e.g. train_BECAUSE.ids.txt

x: sentence (1st chunk, before discourse marker)
x2: sentence (2nd chunk, after discourse marker)
y: classification (label 0: because ; 1:but)

cause and effect: load in one file

2 pair_iter, one for each task
but wrap both in one if i have time with a flag
"""


# def pair_iter(fnamex, fnamex2, batch_size, num_layers, sort_and_shuffle=True):
#     fdx, fdx2 = open(fnamex), open(fnamex2)
#     batches = []
#
#     while True:
#         if len(batches) == 0:
#             refill(batches, fdx, fdx2, fdy, batch_size, sort_and_shuffle=sort_and_shuffle)
#         if len(batches) == 0:
#             break
#
#         x_tokens, x2_tokens, y_tokens = batches.pop(0)
#         y_tokens = add_sos(y_tokens)
#         x_padded, x2_padded, y_padded = padded(x_tokens, num_layers, question_len), \
#                                         padded(x2_tokens, num_layers, context_len), \
#                                         padded(y_tokens, 1)
#
#         source_tokens = np.array(x_padded).T
#         source_mask = (source_tokens != qa_data.PAD_ID).astype(np.int32)
#         source2_tokens = np.array(x2_padded).T
#         source2_mask = (source2_tokens != qa_data.PAD_ID).astype(np.int32)
#         target_tokens = np.array(y_padded).T  # (time_step, batch_size)
#         target_mask = (target_tokens != qa_data.PAD_ID).astype(np.int32)
#
#         yield (source_tokens, source_mask, source2_tokens, source2_mask, target_tokens, target_mask)
#
#     return


# def refill(batches, fd_because, fd_but, batch_size, sort_and_shuffle=True):
#     # context_len restricts samples smaller than context_len
#     line_pairs = []
#     linex, linex2, liney = fdx.readline(), fdx2.readline(), fdy.readline()

#     while linex and linex2 and liney:
#         x_tokens, x2_tokens, y_tokens = tokenize(linex), tokenize(linex2), tokenize(liney)

#         if len(x_tokens) < FLAGS.question_len and len(y_tokens) < FLAGS.max_seq_len \
#                 and len(x2_tokens) <= FLAGS.max_seq_len:
#             line_pairs.append((x_tokens, x2_tokens, y_tokens))
#         if len(line_pairs) == batch_size * 160:
#             break
#         linex, linex2, liney = fdx.readline(), fdx2.readline(), fdy.readline()

#     if sort_and_shuffle:
#         line_pairs = sorted(line_pairs, key=lambda e: len(e[0]))

#     for batch_start in xrange(0, len(line_pairs), batch_size):
#         x_batch, x2_batch, y_batch = zip(*line_pairs[batch_start:batch_start + batch_size])

#         batches.append((x_batch, x2_batch, y_batch))

#     if sort_and_shuffle:
#         random.shuffle(batches)
#     return


# save as pickle file
# list of lists
def winnograd_batches(data_dir, split, vocab, batch_size, cache_filename, 
                      shuffle=True):
    batches = []

    fname_because = pjoin(data_dir, split + "_BECAUSE.ids.txt")
    fd_because = open(fname_because)

    line_pairs = []

    line_because = fd_because.readline()
    winograd_label = 0

    while line_because:
        because_tokens = tokenize(line_because)

        because_id = vocab["because"]
        of_id = vocab["of"]

        # check if the discourse relations are even in the sentences
        # (they're supposed to be, but apparently they're not, in practice??!)
        # idunno why this is.
        if because_id in because_tokens:
            index_of_because = because_tokens.index(because_id)
            # grab sentence chunk before 'because'
            x1_because_tokens = because_tokens[:index_of_because]
            # second chunk should not start with 'of'
            if (of_id in because_tokens) and (because_tokens.index(of_id) == index_of_because + 1):
                because_start_of_next_chunk = index_of_because + 2
            else:
                because_start_of_next_chunk = index_of_because + 1
            x2_because_tokens = because_tokens[because_start_of_next_chunk:]

            sentence = " ".join([rev_vocab[w] for w in because_tokens])

            # exclude sentences that are too long
            if len(x1_because_tokens) <= FLAGS.max_seq_len \
                    and len(x2_because_tokens) <= FLAGS.max_seq_len:
                new_pairs = [
                    (x1_because_tokens, x2_because_tokens, 0, winograd_label, sentence)
                ];
                line_pairs += new_pairs;

        line_because = fd_because.readline()
        # switch off between correct and incorrect
        winograd_label = 1-winograd_label

    if shuffle:
        np.random.shuffle(line_pairs)

    for batch_start in xrange(0, len(line_pairs), batch_size):
        batch_end = batch_start + batch_size
        x1_batch, x2_batch, y_batch, winnograd_labels_batch, line_because_batch = zip(*line_pairs[batch_start:batch_end])

        batches.append((x1_batch, x2_batch, y_batch, winnograd_labels_batch, line_because_batch))

    if shuffle:
        np.random.shuffle(batches)

    pickle.dump(batches, open(cache_filename, "wb"))

    return batches

def winograd_pair_iter(data_dir, vocab, batch_size, shuffle=True):
    """
    Create batches of inputs for but/because classifier, but getting its
    predictions for Winograd schema sentences

    Keyword arguments:
    data_dir -- data directory
    vocab -- a dict from words to their ids in vocab
    batch_size -- number of sentences per batch
    shuffle -- we don't want to shuffle the validation and test sets.

    """

    cache_filename = pjoin(data_dir, "valid_" + str(batch_size) + ".pkl")
    ## if file exists,
    if os.path.isfile(cache_filename):
        logging.info("restoring old batches")
        batches = pickle.load(open(cache_filename, 'rb'))
    else:
        ## fill up batches from pickle file, or make pickle file if necessary
        logging.info("generate new batches")
        batches = winnograd_batches(data_dir, "valid", vocab, batch_size,
                                    cache_filename, shuffle=shuffle)

    while True:
        if len(batches) == 0:
            # stopping condition, when batches is empty
            break

        x_tokens, x2_tokens, y, winograd_label, line_because = batches.pop(0)
        # pad sentence chunks
        x_padded, x2_padded = padded(x_tokens), padded(x2_tokens)

        # first part of sentence (before discourse marker)
        source_tokens = np.array(x_padded)
        source_mask = (source_tokens != data.PAD_ID).astype(np.int32)

        # second part of sentence (after discourse marker)
        source2_tokens = np.array(x2_padded)
        source2_mask = (source2_tokens != data.PAD_ID).astype(np.int32)

        # class ID for this sentence (either 0 for because or 1 for but)
        target_class = y

        yield (source_tokens, source_mask, source2_tokens, source2_mask,
               target_class, winograd_label, line_because)

    return

def but_detector_pair_iter(data_dir, split, vocab, batch_size, shuffle=True):
    """Create batches of inputs for but/because classifier.

    Keyword arguments:
    data_dir -- data directory
    split -- train, valid, or test
    vocab -- a dict from words to their ids in vocab
    batch_size -- number of sentences per batch
    shuffle -- we don't want to shuffle the validation and test sets.

    """

    cache_filename = pjoin(data_dir, split + "_" + str(batch_size) + ".pkl")
    ## if file exists,
    if os.path.isfile(cache_filename):
        logging.info("restoring old batches")
        batches = pickle.load(open(cache_filename, 'rb'))
    else:
        ## fill up batches from pickle file, or make pickle file if necessary
        logging.info("generate new batches")
        batches = but_detector_data_precache(data_dir, split, vocab, batch_size,
                                             cache_filename, shuffle=shuffle)

    while True:
        if len(batches) == 0:
            # stopping condition, when batches is empty
            break

        x_tokens, x2_tokens, y = batches.pop(0)
        # pad sentence chunks
        x_padded, x2_padded = padded(x_tokens), padded(x2_tokens)

        # first part of sentence (before discourse marker)
        source_tokens = np.array(x_padded)
        source_mask = (source_tokens != data.PAD_ID).astype(np.int32)
        # second part of sentence (after discourse marker)
        source2_tokens = np.array(x2_padded)
        source2_mask = (source2_tokens != data.PAD_ID).astype(np.int32)
        # class ID for this sentence (either 0 for because or 1 for but)
        target_class = y

        yield (source_tokens, source_mask, source2_tokens, source2_mask,
               target_class)

    return


# save as pickle file
# list of lists
def but_detector_data_precache(data_dir, split, vocab,
                               batch_size, cache_filename, shuffle=True):
    batches = []

    fname_because = pjoin(data_dir, split + "_BECAUSE.ids.txt")
    fname_but = pjoin(data_dir, split + "_BUT.ids.txt")

    fd_because, fd_but = open(fname_because), open(fname_but)

    line_pairs = []
    discourse_markers = ["because", "but"]

    line_because = fd_because.readline()
    line_but = fd_but.readline()

    while line_because and line_but:
        because_tokens, but_tokens = tokenize(line_because), tokenize(line_but)

        because_id = vocab["because"]
        of_id = vocab["of"]
        but_id = vocab["but"]

        # check if the discourse relations are even in the sentences
        # (they're supposed to be, but apparently they're not, in practice??!)
        # idunno why this is.
        if because_id in because_tokens and but_id in but_tokens:
            index_of_because = because_tokens.index(because_id)
            # grab sentence chunk before 'because'
            x1_because_tokens = because_tokens[:index_of_because]
            # second chunk should not start with 'of'
            if (of_id in because_tokens) and (because_tokens.index(of_id) == index_of_because + 1):
                because_start_of_next_chunk = index_of_because + 2
            else:
                because_start_of_next_chunk = index_of_because + 1
            x2_because_tokens = because_tokens[because_start_of_next_chunk:]

            # grab sentence chunks for 'but'
            index_of_but = but_tokens.index(but_id)
            x1_but_tokens = but_tokens[:index_of_but]
            x2_but_tokens = but_tokens[index_of_but + 1:]

            # exclude sentences that are too long
            if len(x1_because_tokens) <= FLAGS.max_seq_len \
                    and len(x2_because_tokens) <= FLAGS.max_seq_len \
                    and len(x1_but_tokens) <= FLAGS.max_seq_len \
                    and len(x2_but_tokens) <= FLAGS.max_seq_len:
                new_pairs = [
                    (x1_because_tokens, x2_because_tokens, 0),
                    (x1_but_tokens, x2_but_tokens, 1)
                ];
                if shuffle:
                    np.random.shuffle(new_pairs);
                line_pairs += new_pairs;

        line_because, line_but = fd_because.readline(), fd_but.readline()

    if shuffle:
        np.random.shuffle(line_pairs)

    for batch_start in xrange(0, len(line_pairs), batch_size):
        batch_end = batch_start + batch_size
        x1_batch, x2_batch, y_batch = zip(*line_pairs[batch_start:batch_end])

        batches.append((x1_batch, x2_batch, y_batch))

    if shuffle:
        np.random.shuffle(batches)

    pickle.dump(batches, open(cache_filename, "wb"))

    return batches


def cause_effect_pair_iter(fname_because, vocab, batch_size, shuffle=True):
    """Create batches of inputs for but/because classifier.

    Keyword arguments:
    fname_because -- name of "because" data file (e.g. train_BECAUSE.ids.txt)
    vocab -- a dict from words to their ids in vocab
    batch_size -- number of sentences per batch
    shuffle -- flag to shuffle the examples completely

    """
    fd_because = open(fname_because)
    batches = []

    while True:
        if len(batches) == 0:
            # initialize patches
            cause_effect_refill(batches, fd_because, vocab,
                                batch_size, shuffle=shuffle)
        if len(batches) == 0:
            # stopping condition, when batches is empty even after refill
            break

        x_tokens, x2_tokens, y = batches.pop(0)
        # pad sentence chunks
        x_padded, x2_padded = padded(x_tokens), padded(x2_tokens)

        # first part of sentence (before discourse marker)
        source_tokens = np.array(x_padded)
        source_mask = (source_tokens != data.PAD_ID).astype(np.int32)
        # second part of sentence (after discourse marker)
        source2_tokens = np.array(x2_padded)
        source2_mask = (source2_tokens != data.PAD_ID).astype(np.int32)
        # class ID for this sentence (either 0 for because or 1 for but)
        target_class = y

        yield (source_tokens, source_mask, source2_tokens, source2_mask,
               target_class)

    return


def cause_effect_refill(batches, fd_because, vocab, batch_size,
                        shuffle=True):
    """Mutates batches list to fill with tuples of sentence chunks and class id

    Keyword arguments:
    batches -- the batches list to mutate
    fd_because -- loaded "because" sentences
    vocab -- a dict from words to their ids in vocab
    fd_but -- loaded "but" sentences
    batch_size -- number of sentences per batch
    shuffle -- don't shuffle valdiation and test sets

    """
    line_pairs = []

    line = fd_because.readline()

    while line:
        because_tokens = tokenize(line)

        because_id = vocab["because"]
        of_id = vocab["of"]

        if because_id in because_tokens:
            index_of_because = because_tokens.index(because_id)
            effect_tokens = because_tokens[:index_of_because]

            if of_id in because_tokens:
                index_of_of = because_tokens.index(of_id)
                if index_of_of == index_of_because + 1:
                    cause_tokens = because_tokens[index_of_because + 2:]
                else:
                    cause_tokens = because_tokens[index_of_because + 1:]
            else:
                cause_tokens = because_tokens[index_of_because + 1:]

        # exclude sentences that are too long
        if len(cause_tokens) <= FLAGS.max_seq_len and \
            FLAGS.max_seq_len >= len(effect_tokens) > 0:
            # 0 is incorrect, 1 is correct
            if effect_tokens[-1] == 4:
                effect_tokens = effect_tokens[:-1]

            if np.random.randint(0, 2):
                line_pairs.append((cause_tokens, effect_tokens, 0))
            else:
                line_pairs.append((effect_tokens, cause_tokens, 1))

        # only grab 160 batches at once
        if len(line_pairs) == batch_size * 5:
            break

        line = fd_because.readline()

    if shuffle:
        np.random.shuffle(line_pairs)

    for batch_start in xrange(0, len(line_pairs), batch_size):
        batch_end = batch_start + batch_size
        x1_batch, x2_batch, y_batch = zip(*line_pairs[batch_start:batch_end])

        batches.append((x1_batch, x2_batch, y_batch))

    if shuffle:
        np.random.shuffle(batches)

    return


def padded(tokens, batch_pad=0):
    maxlen = max(map(lambda x: len(x), tokens)) if batch_pad == 0 else batch_pad
    return map(lambda token_list: token_list + [data.PAD_ID] * (maxlen - len(token_list)), tokens)


# data_dir, split, vocab, batch_size
if __name__ == '__main__':
    # print(next(cause_effect_pair_iter("data/ptb/train_BECAUSE.ids.txt",
    #                                   {"because": 10, "but": 5, "of": 3}, 20)))
    print(next(winograd_pair_iter(
        "data/ptb/",
        {"because": 10, "but": 5, "of": 3},
        20
    )))
