from __future__ import absolute_import, division, print_function
from copy import deepcopy
import time
import os
import sys
import logging

import tensorflow as tf
import numpy as np
import data

from os.path import join as pjoin
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("state_size", 512, "hidden dimension")
tf.app.flags.DEFINE_integer("layers", 1, "number of hidden layers")
tf.app.flags.DEFINE_integer("epochs", 8, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("embedding_size", 300, "dimension of GloVE vector to use")
tf.app.flags.DEFINE_integer("max_seq_len", 50, "max sequence length")
tf.app.flags.DEFINE_integer("learning_rate_decay_epoch", 1, "Learning rate starts decaying after this epoch.")
tf.app.flags.DEFINE_float("dropout", 0.2, "probability of dropping units")
tf.app.flags.DEFINE_integer("batch_size", 300, "batch size")
tf.app.flags.DEFINE_integer("seed", 123, "random seed to use")
tf.app.flags.DEFINE_float("init_scale", 0.1, "scale for random initialization")
tf.app.flags.DEFINE_string("rnn", "lstm", "lstm/gru architecture choice")
tf.app.flags.DEFINE_string("opt", "adam", "adam/sgd, the optimizer to use")
tf.app.flags.DEFINE_float("learning_rate", 0.003, "initial learning rate")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.8, "amount to decrease learning rate")
tf.app.flags.DEFINE_integer("keep", 5, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_integer("print_every", 5, "How many iterations to do per print.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_string("run_dir", "sandbox", "directory to store experiment outputs")
tf.app.flags.DEFINE_string("dataset", "wikitext-103", "ptb/wikitext-103 select the dataset to use")
tf.app.flags.DEFINE_string("restore_checkpoint", None, "checkpoint file to restore model parameters from")
tf.app.flags.DEFINE_integer("restore_epoch", 0, "the epoch of checkpoint file")
tf.app.flags.DEFINE_boolean("dev", False, "if flag true, will run on dev dataset in a pure testing mode")
tf.app.flags.DEFINE_boolean("temp_max", False, "if flag true, will use Temporal Max Pooling")
tf.app.flags.DEFINE_boolean("temp_mean", False, "if flag true, will use Temporal Mean Pooling")
tf.app.flags.DEFINE_boolean("correct_example", False, "if flag false, will print error, true will print out success")
tf.app.flags.DEFINE_boolean("snli", False, "if flag True, the classifier will train on SNLI")
tf.app.flags.DEFINE_boolean("abs", False, "if flag True, the classifier will train on absolute difference vec op")
tf.app.flags.DEFINE_boolean("concat", False, "if flag True, bidirectional does concatenation not average")
tf.app.flags.DEFINE_integer("num_examples", 30, "enter the best epoch to use")
tf.app.flags.DEFINE_string("prefix", "", "provide the prefix to the data/glove embeddings, used for Deep Clusters")

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

def padded(tokens, batch_pad=0):
    maxlen = max(map(lambda x: len(x), tokens)) if batch_pad == 0 else batch_pad
    return map(lambda token_list: token_list + [data.PAD_ID] * (maxlen - len(token_list)), tokens)

def pair_iter(q, batch_size, inp_len, query_len):
    # use inp_len, query_len to filter list
    batched_seq1 = []
    batched_seq2 = []
    batched_label = []
    iter_q = q[:]

    while len(iter_q) > 0:
        while len(batched_seq1) < batch_size and len(iter_q) > 0:
            pair = iter_q.pop(0)
            if len(pair[0]) <= inp_len and len(pair[1]) <= query_len:
                batched_seq1.append(pair[0])
                batched_seq2.append(pair[1])
                batched_label.append(pair[2])

        padded_input = np.array(padded(batched_seq1), dtype=np.int32)
        input_mask = (padded_input != data.PAD_ID).astype(np.int32)
        padded_query = np.array(padded(batched_seq2), dtype=np.int32)
        query_mask = (padded_query != data.PAD_ID).astype(np.int32)
        labels = np.array(batched_label, dtype=np.int32)

        yield padded_input, input_mask, padded_query, query_mask, labels
        batched_seq1, batched_seq2, batched_label = [], [], []

class Encoder(object):
    def __init__(self, size, num_layers):
        self.size = size
        self.keep_prob = tf.placeholder(tf.float32)

        if FLAGS.rnn == "lstm":
            cell = rnn_cell.BasicLSTMCell(self.size)
            state_is_tuple = True
        else:
            cell = rnn_cell.GRUCell(self.size)
            state_is_tuple = False

        cell = DropoutWrapper(cell, input_keep_prob=self.keep_prob, seed=123)
        self.encoder_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=state_is_tuple)

    # could consider instead of averaging, I concatenate
    def encode(self, inputs, masks, reuse=False, scope_name="", temp_max=False):
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
                # (batch_size, T, hidden_size)
                out = fw_out + bw_out

            # before we are using state_is_tuple=True, meaning we only chose top layer
            # now we choose both so layer 1 and layer 2 will have a difference
            # this is extracting the last hidden states
            if FLAGS.rnn == "gru":
                encoder_outputs = tf.add(output_state_fw, output_state_bw)  # used to have [0][1]
            else:
                # last layer [-1], hidden state [1]
                # this works with multilayer
                if temp_max:
                    max_forward = tf.reduce_max(fw_out, axis=1)
                    max_backward = tf.reduce_max(bw_out, axis=1)
                    if not FLAGS.concat:
                        encoder_outputs = max_forward + max_backward
                    else:
                        encoder_outputs = tf.concat(1, [max_forward, max_backward])
                elif FLAGS.temp_mean:
                    mean_forward = tf.reduce_mean(fw_out, axis=1)
                    mean_backward = tf.reduce_mean(bw_out, axis=1)
                    if not FLAGS.concat:
                        encoder_outputs = mean_forward + mean_backward
                    else:
                        encoder_outputs = tf.concat(1, [mean_forward, mean_backward])
                else:
                    if not FLAGS.concat:
                        encoder_outputs = tf.add(output_state_fw[-1][1], output_state_bw[-1][1])
                    else:
                        encoder_outputs = tf.concat(1, [output_state_fw[-1][1], output_state_bw[-1][1]])

        return out, encoder_outputs


class AttentionEncoder(object):
    def __init__(self, size, num_layers):
        self.size = size
        self.keep_prob = tf.placeholder(tf.float32)


class SequenceClassifier(object):
    def __init__(self, encoder, flags, vocab_size, vocab, rev_vocab, label_size, embed_path,
                 optimizer="adam", is_training=True):

        self.max_seq_len = flags.max_seq_len
        self.encoder = encoder
        self.embed_path = embed_path
        self.vocab = vocab
        self.rev_vocab = rev_vocab
        self.vocab_size = vocab_size
        self.flags = flags
        self.label_size = label_size

        logging.info("label size is: {}".format(self.label_size))

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
        seqA_w_matrix, seqA_c_vec = self.encoder.encode(self.seqA_inputs, self.seqA_mask, temp_max=FLAGS.temp_max)
        seqB_w_matrix, seqB_c_vec = self.encoder.encode(self.seqB_inputs, self.seqB_mask, reuse=True , temp_max=FLAGS.temp_max)

        self.seqA_rep = seqA_c_vec
        self.seqB_rep = seqB_c_vec

        # for now we just use context vector
        # we create additional perspectives

        # seqA_c_vec: (batch_size, hidden_size)
        persA_B_mul = seqA_c_vec * seqB_c_vec
        persA_B_sub = seqA_c_vec - seqB_c_vec
        persA_B_avg = (seqA_c_vec + seqB_c_vec) / 2.0

        # logits is [batch_size, label_size]
        if FLAGS.abs:
            persA_B_sub = tf.abs(seqA_c_vec - seqB_c_vec)
            self.logits = rnn_cell._linear([seqA_c_vec, seqB_c_vec, persA_B_mul, persA_B_sub],
                                           self.label_size, bias=True)
        else:
            self.logits = rnn_cell._linear([seqA_c_vec, seqB_c_vec, persA_B_mul, persA_B_sub, persA_B_avg],
                                       self.label_size, bias=True)

    def optimize(self, session, seqA_tokens, seqA_mask, seqB_tokens, seqB_mask, labels):
        input_feed = {}
        input_feed[self.seqA] = seqA_tokens
        input_feed[self.seqA_mask] = seqA_mask
        input_feed[self.seqB] = seqB_tokens
        input_feed[self.seqB_mask] = seqB_mask
        input_feed[self.labels] = labels

        input_feed[self.encoder.keep_prob] = self.keep_prob_config

        output_feed = [self.updates, self.logits, self.gradient_norm, self.loss, self.param_norm, self.seqA_rep]

        outputs = session.run(output_feed, input_feed)

        return outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]

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

    def extract_sent(self, positions, sent):
        list_sent = sent.tolist()
        extracted_sent = []
        for i in range(sent.shape[0]):
            if positions[i]:
                extracted_sent.append(list_sent[i])
        return extracted_sent

    def get_multiclass_accuracy(self, preds, labels):
        label_cat = range(self.label_size)
        labels_accu = {}

        for la in label_cat:
            # for each label, we get the index of the correct labels
            idx_of_cat = labels == la
            cat_preds = preds[idx_of_cat]
            if cat_preds.size != 0:
                accu = np.mean(cat_preds == la)
                labels_accu[la] = [accu]
            else:
                labels_accu[la] = []

        return labels_accu

    def cumulate_multiclass_accuracy(self, total_accu, labels_accu):
        for k, v in labels_accu.iteritems():
            total_accu[k].extend(v)

    def get_mean_multiclass_accuracy(self, total_accu):
        for k, v in total_accu.iteritems():
            total_accu[k] = np.mean(total_accu[k])

    def but_because_validate(self, session, q, label_tokens, dev=False):
        # class_label: [because, but, ...]
        valid_costs, valid_accus = [], []
        valid_preds, valid_labels = [], []
        # valid_sent1, valid_sent2 = [], []

        total_labels_accu = None

        for seqA_tokens, seqA_mask, seqB_tokens, \
                seqB_mask, labels in pair_iter(q, self.flags.batch_size, self.max_seq_len, self.max_seq_len):
            cost, logits = self.test(session, seqA_tokens, seqA_mask, seqB_tokens, seqB_mask, labels)
            valid_costs.append(cost)
            accu = np.mean(np.argmax(logits, axis=1) == labels)

            preds = np.argmax(logits, axis=1)

            labels_accu = self.get_multiclass_accuracy(preds, labels)
            if total_labels_accu is None:
                total_labels_accu = labels_accu
            else:
                self.cumulate_multiclass_accuracy(total_labels_accu, labels_accu)

            # wrong_preds = np.extract(positions, preds)
            # print(wrong_preds)
            # print(np.extract(positions, labels))

            # print()
            # print(preds.tolist())
            # print(list(labels))

            # TODO: if we need to print out examples again, just git revert,
            # TODO: check an earlier version of this file

            valid_preds.extend(preds.tolist())
            valid_labels.extend(labels.tolist())

            valid_accus.append(accu)

        valid_accu = sum(valid_accus) / float(len(valid_accus))
        valid_cost = sum(valid_costs) / float(len(valid_costs))

        self.get_mean_multiclass_accuracy(total_labels_accu)
        multiclass_accu_msg = ''
        for k, v in total_labels_accu.iteritems():
            multiclass_accu_msg += label_tokens[k] + ": " + str(v) + " "

        logging.info(multiclass_accu_msg)

        if dev:
            return valid_cost, valid_accu, valid_preds, valid_labels

        return valid_cost, valid_accu

    def setup_cause_effect(self):
        # seqA: but, seqB: because, this will learn to differentiate them
        seqA_w_matrix, seqA_c_vec = self.encoder.encode(self.seqA_inputs, self.seqA_mask)
        seqB_w_matrix, seqB_c_vec = self.encoder.encode(self.seqB_inputs, self.seqB_mask, reuse=True)

        self.seqA_rep = seqA_c_vec
        self.seqB_rep = seqB_c_vec

        # for now we just use context vector
        # we create additional perspectives

        # seqA_c_vec: (batch_size, hidden_size)
        persA_B_mul = seqA_c_vec * seqB_c_vec
        persA_B_sub = seqA_c_vec - seqB_c_vec
        persA_B_avg = (seqA_c_vec + seqB_c_vec) / 2.0

        # logits is [batch_size, label_size]
        self.logits = rnn_cell._linear([seqA_c_vec, seqB_c_vec, persA_B_mul, persA_B_sub, persA_B_avg],
                                       self.label_size, bias=True)

    def detokenize_batch(self, sent):
        # sent: (N, sentence_padded_length)
        def detok_sent(sent):
            outsent = ''
            for t in sent:
                if t > 0:  # only take out pad, but not unk
                    outsent += self.rev_vocab[t] + " "
            return outsent
        return [detok_sent(s) for s in sent]

    def but_because_dev_test(self, session, q, save_train_dir, label_tokens):
        ## Checkpoint
        import csv

        # checkpoint_path = os.path.join(save_train_dir, "dis.ckpt")
        #
        # logging.info("restore model from best epoch %d" % best_epoch)
        # self.saver.restore(session, checkpoint_path + ("-%d" % best_epoch))

        # load into the "dev" files
        # , valid_sent1, valid_sent2
        test_cost, test_accu, test_preds, test_labels= self.but_because_validate(session, q, label_tokens, dev=True)

        logging.info("Final test cost: %f test accu: %f" % (test_cost, test_accu))

        examples = 0

        with open(pjoin(save_train_dir, 'confusion_test.csv'), 'wb') as csvfile:
            fieldnames = ['preds', 'labels']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for pair in zip(test_preds, test_labels):
                writer.writerow({'preds': pair[0], 'labels': pair[1]})

            # print("true label: {}, predicted: {}, sent1: {}, sent2: {}".format(pair[1], pair[0], pair[2], pair[3]))
            # examples += 1
            # if examples >= FLAGS.num_examples:
            #     break

        sys.stdout.flush()

    def but_because_train(self, session, q_train, q_valid, q_test, label_tokens, curr_epoch, num_epochs, save_train_dirs):

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        lr = FLAGS.learning_rate
        epoch = curr_epoch
        best_epoch = 0
        previous_losses = []
        valid_accus = []
        exp_cost = None
        exp_norm = None

        while num_epochs == 0 or epoch < num_epochs:
            epoch += 1
            current_step = 0

            ## Train
            epoch_tic = time.time()
            for seqA_tokens, seqA_mask, seqB_tokens, \
                seqB_mask, labels in pair_iter(q_train, self.flags.batch_size, self.max_seq_len, self.max_seq_len):
                # Get a batch and make a step.
                tic = time.time()

                logits, grad_norm, cost, param_norm, seqA_rep = self.optimize(session, seqA_tokens, seqA_mask,
                                                            seqB_tokens, seqB_mask, labels)

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
            checkpoint_path = os.path.join(save_train_dirs, "dis.ckpt")

            ## Validate
            valid_cost, valid_accu = self.but_because_validate(session, q_valid, label_tokens)

            logging.info("Epoch %d Validation cost: %f validation accu: %f epoch time: %f" % (epoch, valid_cost,
                                                                                              valid_accu,
                                                                                              epoch_toc - epoch_tic))

            # if epoch >= self.flags.learning_rate_decay_epoch:
            #     lr *= FLAGS.learning_rate_decay
            #     logging.info("Annealing learning rate at epoch {} to {}".format(epoch, lr))
            #     session.run(self.learning_rate_decay_op)

            # only do accuracy
            if len(previous_losses) >= 1 and valid_accu < max(valid_accus):
                lr *= FLAGS.learning_rate_decay
                logging.info("Annealing learning rate at epoch {} to {}".format(epoch, lr))
                session.run(self.learning_rate_decay_op)

                logging.info("validation cost trigger: restore model from epoch %d" % best_epoch)
                self.saver.restore(session, checkpoint_path + ("-%d" % best_epoch))
            else:
                previous_losses.append(valid_cost)
                best_epoch = epoch
                self.saver.save(session, checkpoint_path, global_step=epoch)

            valid_accus.append(valid_accu)

        logging.info("restore model from best epoch %d" % best_epoch)
        logging.info("best validation accuracy: %f" % valid_accus[best_epoch - 1])
        self.saver.restore(session, checkpoint_path + ("-%d" % best_epoch))

        # after training, we test this thing
        ## Test
        test_cost, test_accu = self.but_because_validate(session, q_test, label_tokens)
        logging.info("Final test cost: %f test accu: %f" % (test_cost, test_accu))

        logging.info("Saving confusion matrix csv")
        self.but_because_dev_test(session, q_test, FLAGS.run_dir, label_tokens)

        sys.stdout.flush()