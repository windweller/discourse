from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange
import tensorflow as tf
import random

import data

FLAGS = tf.app.flags.FLAGS

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

def but_detector_pair_iter(fname_because, fname_but, relation_vocab, batch_size,
                           num_layers, sort_and_shuffle=True):
    """Create batches of inputs for but/because classifier.

    Keyword arguments:
    fname_because -- name of "because" data file (e.g. train_BECAUSE.ids.txt)
    fname_but -- name of "but" data file (e.g. ptb/train_BUT.ids.txt)
    relation_vocab -- a dict from discourse markers to their ids in vocab
    batch_size -- number of sentences per batch
    num_layers -- idunno what this is for, but it gets passed into `padded`
    sort_and_shuffle -- idunno what this is for

    """
    fd_because, fd_but = open(fname_because), open(fname_but)
    batches = []

    while True:
        if len(batches) == 0:
            # initialize patches
            but_detector_refill(batches, fd_because, fd_but, relation_vocab,
                                batch_size, sort_and_shuffle=sort_and_shuffle)
        if len(batches) == 0:
            # stopping condition, when batches is empty again
            break

        x_tokens, x2_tokens, y = batches.pop(0)
        # pad sentence chunks
        # idunno if this should use FLAGS or something else.
        # the orig here use question_length or something.
        x_padded, x2_padded = padded(x_tokens, FLAGS.max_seq_len), \
                              padded(x2_tokens, FLAGS.max_seq_len)

        # first part of sentence (before discourse marker)
        source_tokens = np.array(x_padded).T
        source_mask = (source_tokens != data.PAD_ID).astype(np.int32)
        # second part of sentence (after discourse marker)
        source2_tokens = np.array(x2_padded).T
        source2_mask = (source2_tokens != data.PAD_ID).astype(np.int32)
        # class ID for this sentence (either 0 for because or 1 for but)
        target_class = y

        yield (source_tokens, source_mask, source2_tokens, source2_mask,
               target_class)

    return

def but_detector_refill(batches, fd_because, fd_but, relation_vocab, batch_size,
                        shuffle=True):
    """Mutates batches list to fill with tuples of sentence chunks and class id

    Keyword arguments:
    batches -- the batches list to mutate
    fd_because -- loaded "because" sentences
    relation_vocab -- a dict from discourse markers to their ids in vocab
    fd_but -- loaded "but" sentences
    batch_size -- number of sentences per batch
    shuffle -- flag to shuffle the examples completely

    """
    line_pairs = []
    fds = {"because": fd_because, "but": fd_but}
    discourse_markers = ["because", "but"]

    # accumulate tuples for every sentence from each file
    for target_class in [0,1]:
        discourse_marker = discourse_markers[target_class]
        relation_id_in_vocab = relation_vocab[discourse_marker]
        fd = fds[discourse_marker]
        line = fd.readline()
        while line:
            tokens = tokenize(line)

            # split by relevant discourse relation
            index_of_relation = tokens.index(relation_id_in_vocab)
            x1_tokens = tokens[:index_of_relation]
            x2_tokens = tokens[index_of_relation+1:]

            y = target_class

            # exclude sentences that are too long
            if len(x1_tokens) <= FLAGS.max_seq_len \
                    and len(x2_tokens) <= FLAGS.max_seq_len:
                line_pairs.append((x1_tokens, x2_tokens, y))
            # idunno where the number 160 is coming from
            # why do we want max 160 batches?
            if len(line_pairs) == batch_size * 160:
                break

            line = fd.readline()

    # shuffle order of examples completely
    if shuffle:
        line_pairs = random.shuffle(line_pairs)

    for batch_start in xrange(0, len(line_pairs), batch_size):
        batch_end = batch_start + batch_size
        x1_batch, x2_batch, y_batch = zip(*line_pairs[batch_start:batch_end])

        batches.append((x1_batch, x2_batch, y_batch))

    return


# def refill(batches, fd_because, fd_but, batch_size, sort_and_shuffle=True):
#     # context_len restricts samples smaller than context_len
#     line_pairs = []
#     linex, linex2, liney = fdx.readline(), fdx2.readline(), fdy.readline()
#
#     while linex and linex2 and liney:
#         x_tokens, x2_tokens, y_tokens = tokenize(linex), tokenize(linex2), tokenize(liney)
#
#         if len(x_tokens) < FLAGS.question_len and len(y_tokens) < FLAGS.max_seq_len \
#                 and len(x2_tokens) <= FLAGS.max_seq_len:
#             line_pairs.append((x_tokens, x2_tokens, y_tokens))
#         if len(line_pairs) == batch_size * 160:
#             break
#         linex, linex2, liney = fdx.readline(), fdx2.readline(), fdy.readline()
#
#     if sort_and_shuffle:
#         line_pairs = sorted(line_pairs, key=lambda e: len(e[0]))
#
#     for batch_start in xrange(0, len(line_pairs), batch_size):
#         x_batch, x2_batch, y_batch = zip(*line_pairs[batch_start:batch_start + batch_size])
#
#         batches.append((x_batch, x2_batch, y_batch))
#
#     if sort_and_shuffle:
#         random.shuffle(batches)
#     return


def padded(tokens, batch_pad=0):
  maxlen = max(map(lambda x: len(x), tokens)) if batch_pad == 0 else batch_pad
  return map(lambda token_list: token_list + [data.PAD_ID] * (maxlen - len(token_list)), tokens)
