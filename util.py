from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange
import tensorflow as tf
import random

FLAGS = tf.app.flags.FLAGS

def tokenize(string):
    return [int(s) for s in string.split()]


"""
fnamex: e.g. train_BECAUSE.ids.txt

x: sentence (1st chunk, before discourse marker)
x2: sentence (2nd chunk, after discourse marker)
y: classification (label 0: because ; 1:but)

cause and effect: load in one file

"""
def pair_iter(fnamex, fnamex2, batch_size, num_layers, sort_and_shuffle=True):
    fdx, fdx2 = open(fnamex), open(fnamex2)
    batches = []

    while True:
        if len(batches) == 0:
            refill(batches, fdx, fdx2, fdy, batch_size, sort_and_shuffle=sort_and_shuffle)
        if len(batches) == 0:
            break

        x_tokens, x2_tokens, y_tokens = batches.pop(0)
        y_tokens = add_sos(y_tokens)
        x_padded, x2_padded, y_padded = padded(x_tokens, num_layers, question_len), \
                                        padded(x2_tokens, num_layers, context_len), \
                                        padded(y_tokens, 1)

        source_tokens = np.array(x_padded).T
        source_mask = (source_tokens != qa_data.PAD_ID).astype(np.int32)
        source2_tokens = np.array(x2_padded).T
        source2_mask = (source2_tokens != qa_data.PAD_ID).astype(np.int32)
        target_tokens = np.array(y_padded).T  # (time_step, batch_size)
        target_mask = (target_tokens != qa_data.PAD_ID).astype(np.int32)

        yield (source_tokens, source_mask, source2_tokens, source2_mask, target_tokens, target_mask)

    return


def refill(batches, fdx, fdx2, fdy, batch_size, sort_and_shuffle=True):
    # context_len restricts samples smaller than context_len
    line_pairs = []
    linex, linex2, liney = fdx.readline(), fdx2.readline(), fdy.readline()

    while linex and linex2 and liney:
        x_tokens, x2_tokens, y_tokens = tokenize(linex), tokenize(linex2), tokenize(liney)

        if len(x_tokens) < FLAGS.question_len and len(y_tokens) < FLAGS.max_seq_len \
                and len(x2_tokens) <= FLAGS.max_seq_len:
            line_pairs.append((x_tokens, x2_tokens, y_tokens))
        if len(line_pairs) == batch_size * 160:
            break
        linex, linex2, liney = fdx.readline(), fdx2.readline(), fdy.readline()

    if sort_and_shuffle:
        line_pairs = sorted(line_pairs, key=lambda e: len(e[0]))

    for batch_start in xrange(0, len(line_pairs), batch_size):
        x_batch, x2_batch, y_batch = zip(*line_pairs[batch_start:batch_start + batch_size])

        batches.append((x_batch, x2_batch, y_batch))

    if sort_and_shuffle:
        random.shuffle(batches)
    return


def add_sos(tokens):
    return map(lambda token_list: [qa_data.SOS_ID] + token_list, tokens)


def padded(tokens, depth, batch_pad=0):
    maxlen = max(map(lambda x: len(x), tokens)) if batch_pad == 0 else batch_pad
    align = pow(2, depth - 1)
    padlen = maxlen + (align - maxlen) % align
    return map(lambda token_list: token_list + [qa_data.PAD_ID] * (padlen - len(token_list)), tokens)
