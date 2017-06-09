from __future__ import absolute_import, division, print_function
from copy import deepcopy
import time
import os
import sys
import logging

import tensorflow as tf
import numpy as np

from os.path import join as pjoin
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs

from util import but_detector_pair_iter

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("state_size", 256, "hidden dimension")
tf.app.flags.DEFINE_integer("layers", 1, "number of hidden layers")
tf.app.flags.DEFINE_integer("epochs", 8, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("max_seq_len", 35, "number of time steps to unroll for BPTT, also the max sequence length")
tf.app.flags.DEFINE_integer("embedding_size", 100, "dimension of GloVE vector to use")
tf.app.flags.DEFINE_integer("learning_rate_decay_epoch", 1, "Learning rate starts decaying after this epoch.")
tf.app.flags.DEFINE_float("dropout", 0.2, "probability of dropping units")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_integer("seed", 123, "random seed to use")
tf.app.flags.DEFINE_float("init_scale", 0.1, "scale for random initialization")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "initial learning rate")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.8, "amount to decrease learning rate")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_string("run_dir", "sandbox", "directory to store experiment outputs")
tf.app.flags.DEFINE_string("dataset", "ptb", "select the dataset to use")
tf.app.flags.DEFINE_string("task", "but", "choose the task: but/cause")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding")
tf.app.flags.DEFINE_string("restore_checkpoint", None, "checkpoint file to restore model parameters from")

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, num_layers):
        self.size = size
        self.keep_prob = tf.placeholder(tf.float32)
        lstm_cell = rnn_cell.BasicLSTMCell(self.size)
        lstm_cell = DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob)
        self.encoder_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)

    def encode(self, inputs, masks, reuse=False, scope_name=""):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: (time_step, length, size), notice that input is "time-major"
                        instead of "batch-major".
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        with vs.variable_scope(scope_name + "Encoder", reuse=reuse):
            inp = inputs
            mask = masks
            encoder_outputs = None

            with vs.variable_scope("EncoderCell") as scope:
                srclen = tf.reduce_sum(mask, reduction_indices=1)
                (fw_out, bw_out), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(self.encoder_cell,
                                                                                         self.encoder_cell, inp, srclen,
                                                                                         scope=scope, dtype=tf.float32)
                out = fw_out + bw_out

            # this is extracting the last hidden states
            encoder_outputs = tf.add(output_state_fw[0][1], output_state_bw[0][1])

        return out, encoder_outputs

    def dropout(self, inp):
        return tf.nn.dropout(inp, self.keep_prob)


class CNNEncoder(object):
    """
    Partially adapted
    from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py
    """

    def __init__(self):
        pass


class SequenceClassifier(object):
    def __init__(self, encoder, flags, vocab_size, vocab, embed_path, task, optimizer="adam", is_training=True):
        # task: ["but", "cause"]

        batch_size = flags.batch_size
        max_seq_len = flags.max_seq_len
        self.encoder = encoder
        self.embed_path = embed_path
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.flags = flags
        self.label_size = 2

        self.learning_rate = flags.learning_rate
        max_gradient_norm = flags.max_gradient_norm
        keep = flags.keep
        dropout = flags.dropout
        learning_rate_decay = flags.learning_rate_decay

        self.seqA = tf.placeholder(tf.int32, [None, None])
        self.seqB = tf.placeholder(tf.int32, [None, None])
        self.seqA_mask = tf.placeholder(tf.int32, [None, None])
        self.seqB_mask = tf.placeholder(tf.int32, [None, None])

        self.labels = tf.placeholder(tf.int32, [None])

        self.keep_prob_config = 1.0 - dropout
        self.learning_rate = tf.Variable(float(self.learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay)
        self.global_step = tf.Variable(0, trainable=False)

        with tf.device("/cpu:0"):
            embed = tf.constant(np.load(self.embed_path)['glove'], dtype=tf.float32, name="glove",
                                shape=[self.vocab_size, self.flags.embedding_size])
            self.seqA_inputs = tf.nn.embedding_lookup(embed, self.seqA)
            self.seqB_inputs = tf.nn.embedding_lookup(embed, self.seqB)

        # main computation graph is here
        if task == 'but':
            self.setup_but_because()
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.labels))

        if is_training:
            # ==== set up training/updating procedure ====
            params = tf.trainable_variables()
            opt = get_optimizer(optimizer)(self.learning_rate)

            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.gradient_norm = tf.global_norm(gradients)
            self.param_norm = tf.global_norm(params)
            self.updates = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=keep)

    def setup_but_because(self):
        # For Erin: this is the MODEL!!!
        # seqA: but, seqB: because, this will learn to differentiate them
        seqA_w_matrix, seqA_c_vec = self.encoder.encode(self.seqA_inputs, self.seqA_mask)
        seqB_w_matrix, seqB_c_vec = self.encoder.encode(self.seqB_inputs, self.seqB_mask, reuse=True)

        # for now we just use context vector
        # we create additional perspectives

        # seqA_c_vec: (batch_size, hidden_size)
        persA_B_mul = seqA_c_vec * seqB_c_vec
        persA_B_sub = seqA_c_vec - seqB_c_vec
        persA_B_avg = (seqA_c_vec + seqB_c_vec) / 2.0

        # logits is [batch_size, label_size]
        self.logits = rnn_cell._linear([seqA_c_vec, seqB_c_vec, persA_B_mul, persA_B_sub, persA_B_avg],
                                       self.label_size, bias=True)

    def optimize(self, session, because_tokens, because_mask, but_tokens, but_mask, labels):
        input_feed = {}
        input_feed[self.seqA] = because_tokens
        input_feed[self.seqA_mask] = because_mask
        input_feed[self.seqB] = but_tokens
        input_feed[self.seqB_mask] = but_mask
        input_feed[self.labels] = labels

        input_feed[self.encoder.keep_prob] = self.keep_prob_config

        output_feed = [self.updates, self.logits, self.gradient_norm, self.loss, self.param_norm]

        outputs = session.run(output_feed, input_feed)

        return outputs[1], outputs[2], outputs[3], outputs[4]

    def test(self, session, because_tokens, because_mask, but_tokens, but_mask, labels):
        input_feed = {}
        input_feed[self.seqA] = because_tokens
        input_feed[self.seqA_mask] = because_mask
        input_feed[self.seqB] = but_tokens
        input_feed[self.seqB_mask] = but_mask
        input_feed[self.labels] = labels

        input_feed[self.encoder.keep_prob] = 1.

        output_feed = [self.loss, self.logits]

        outputs = session.run(output_feed, input_feed)

        return outputs[0], outputs[1]

    def but_because_validate(self, session, data_dir, split):
        valid_costs, valid_accus = [], []
        for because_tokens, because_mask, but_tokens, \
            but_mask, labels in but_detector_pair_iter(data_dir, split, self.vocab,
                                                       self.flags.batch_size):
            cost, logits = self.test(session, because_tokens, because_mask, but_tokens, but_mask, labels)
            valid_costs.append(cost)
            accu = np.mean(np.argmax(logits, axis=1) == labels)
            valid_accus.append(accu)

        valid_accu = sum(valid_accus) / float(len(valid_accus))
        valid_cost = sum(valid_costs) / float(len(valid_costs))
        return valid_cost, valid_accu

    def setup_cause_effect(self):
        pass

    def but_because_train(self, session, but_train, because_train, but_valid,
                          because_valid, but_test, because_test, curr_epoch, num_epochs, save_train_dir, data_dir):

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        epoch = curr_epoch
        best_epoch = 0
        previous_losses = []
        exp_cost = None
        exp_norm = None

        lr = FLAGS.learning_rate

        while num_epochs == 0 or epoch < num_epochs:
            epoch += 1
            current_step = 0

            ## Train
            epoch_tic = time.time()
            for because_tokens, because_mask, but_tokens, \
                but_mask, labels in but_detector_pair_iter(data_dir, "train", self.vocab,
                                                                     self.flags.batch_size):
                # Get a batch and make a step.
                tic = time.time()

                logits, grad_norm, cost, param_norm = self.optimize(session, because_tokens, because_mask,
                                                            but_tokens, but_mask, labels)

                accu = np.mean(np.argmax(logits, axis=1) == labels)

                toc = time.time()
                iter_time = toc - tic
                current_step += 1

                if not exp_cost:
                    exp_cost = cost
                    exp_norm = grad_norm
                else:
                    exp_cost = 0.99 * exp_cost + 0.01 * cost
                    exp_norm = 0.99 * exp_norm + 0.01 * grad_norm

                if current_step % self.flags.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, cost %f, exp_cost %f, accuracy %f, grad norm %f, param norm %f, batch time %f' %
                        (epoch, current_step, cost, exp_cost, accu, grad_norm, param_norm, iter_time))

            epoch_toc = time.time()

            ## Checkpoint
            checkpoint_path = os.path.join(save_train_dir, "dis.ckpt")

            ## Validate
            valid_cost, valid_accu = self.but_because_validate(session, data_dir, "valid")

            logging.info("Epoch %d Validation cost: %f validation accu: %f epoch time: %f" % (epoch, valid_cost,
                                                                                              valid_accu,
                                                                                              epoch_toc - epoch_tic))

            if len(previous_losses) >= 1 and valid_cost > previous_losses[-1]:
                lr = lr * FLAGS.learning_rate_decay
                logging.info("Epoch %d, learning rate %f" % (epoch, lr))
                session.run(self.learning_rate_decay_op)
                logging.info("validation cost trigger: restore model from epoch %d" % best_epoch)
                self.saver.restore(session, checkpoint_path + ("-%d" % best_epoch))
            else:
                previous_losses.append(valid_cost)
                best_epoch = epoch
                self.saver.save(session, checkpoint_path, global_step=epoch)


        # after training, we test this thing
        ## Test
        test_cost, test_accu = self.but_because_validate(session, data_dir, "test")
        logging.info("Final test cost: %f test accu: %f" % (test_cost, test_accu))

        sys.stdout.flush()