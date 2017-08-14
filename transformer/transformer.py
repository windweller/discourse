from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from transformer import common_attention
from transformer import common_layers

import tensorflow as tf


# set up hparams as this class
class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def add_hparam(self, name, value):
        self[name] = value

def embedding(x, hparams):
    with tf.variable_scope(
            "EmbeddingLayer", default_name="embedding", values=[x], reuse=False):
        embedding_var = tf.get_variable("kernel", [vocab_size, hparams.hidden_size], dtype=tf.float32)
        embedded_inputs = tf.nn.embedding_lookup(embedding_var, x)

    return embedded_inputs

def transformer_prepare_encoder(inputs, hparams):
    """Prepare one shard of the model for the encoder.
    Args:
      inputs: a Tensor. (3d, after flattening)
      target_space: a Tensor.
      hparams: run hyperparameters
    Returns:
      encoder_input: a Tensor, bottom of encoder stack
      encoder_self_attention_bias: a bias tensor for use in encoder self-attention
      encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
        attention
    """
    encoder_input = inputs
    encoder_padding = common_attention.embedding_to_padding(encoder_input)
    ignore_padding = common_attention.attention_bias_ignore_padding(
        encoder_padding)
    encoder_self_attention_bias = ignore_padding
    encoder_decoder_attention_bias = ignore_padding
    if hparams.proximity_bias:
        encoder_self_attention_bias += common_attention.attention_bias_proximal(
            tf.shape(inputs)[1])
    # Append target_space_id embedding to inputs.
    # emb_target_space = common_layers.embedding(
    #     target_space, 32, ishape_static[-1], name="target_space_embedding")
    # emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
    # encoder_input += emb_target_space

    if hparams.pos == "timing":
        encoder_input = common_attention.add_timing_signal_1d(encoder_input)
    return (encoder_input, encoder_self_attention_bias,
            encoder_decoder_attention_bias)


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder"):
    """A stack of transformer layers.
    Args:
      encoder_input: a Tensor
      encoder_self_attention_bias: bias Tensor for self-attention
         (see common_attention.attention_bias())
      hparams: hyperparameters for model
      {num_hidden_layers, attention_key_channels or hidden_size, attention_value_channels or hidden_size,
      num_heads, hidden_size, attention_dropout}
      name: a string
    Returns:
      y: a Tensors
    """
    x = encoder_input
    with tf.variable_scope(name):
        for layer in xrange(hparams.num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = common_attention.multihead_attention(
                        common_layers.layer_preprocess(
                            x, hparams), None, encoder_self_attention_bias,
                        hparams.attention_key_channels or hparams.hidden_size,
                        hparams.attention_value_channels or hparams.hidden_size,
                        hparams.hidden_size, hparams.num_heads, hparams.attention_dropout)
                    x = common_layers.layer_postprocess(x, y, hparams)
                with tf.variable_scope("ffn"):
                    y = transformer_ffn_layer(
                        common_layers.layer_preprocess(x, hparams), hparams)
                    x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it shuold also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


def transformer_ffn_layer(x, hparams):
    """Feed-forward layer in the transformer.
    Args:
      x: a Tensor of shape [batch_size, length, hparams.hidden_size]
      hparams: hyperparmeters for model
    Returns:
      a Tensor of shape [batch_size, length, hparams.hidden_size]
    """
    if hparams.ffn_layer == "conv_hidden_relu":
        return common_layers.conv_hidden_relu(
            x,
            hparams.filter_size,
            hparams.hidden_size,
            dropout=hparams.relu_dropout)
    elif hparams.ffn_layer == "parameter_attention":
        return common_attention.parameter_attention(
            x, hparams.parameter_attention_key_channels or hparams.hidden_size,
               hparams.parameter_attention_value_channels or hparams.hidden_size,
            hparams.hidden_size, hparams.filter_size, hparams.num_heads,
            hparams.attention_dropout)
    elif hparams.ffn_layer == "conv_hidden_relu_with_sepconv":
        return common_layers.conv_hidden_relu(
            x,
            hparams.filter_size,
            hparams.hidden_size,
            kernel_size=(3, 1),
            second_kernel_size=(31, 1),
            padding="LEFT",
            dropout=hparams.relu_dropout)
    else:
        assert hparams.ffn_layer == "none"
        return x


class TransformerEncoder(object):
    """Transformer, encoder only."""

    # https://github.com/tensorflow/tensor2tensor/blob/82cce5208d9b3705c8183feb0079b7c6a69a4790/tensor2tensor/models/slicenet_test.py

    def __init__(self, hparams):
        self._hparams = hparams

    def model_fn(self, features):
        hparams = self._hparams
        inputs = features["inputs"]
        # target_space = features["target_space_id"]  # this is for modality? I think?
        # just pass in : features = {"target_space_id": tf.constant(1, dtype=tf.int32)}

        inputs = embedding(inputs, hparams)
        print(inputs.get_shape())
        inputs = common_layers.flatten4d3d(inputs)  # no need to flatten anymore

        print(inputs.get_shape())

        (encoder_input, encoder_self_attention_bias,
         _) = (transformer_prepare_encoder(inputs, hparams))

        encoder_input = tf.nn.dropout(encoder_input,
                                      1.0 - hparams.layer_prepostprocess_dropout)
        encoder_output = transformer_encoder(encoder_input,
                                             encoder_self_attention_bias, hparams)

        return encoder_output


def transformer_base():
    """Set of hyperparameters."""
    hparams = dotdict()
    hparams.norm_type = "layer"
    hparams.hidden_size = 512
    hparams.batch_size = 4096
    hparams.max_length = 256
    hparams.dropout = 0.0
    hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
    hparams.optimizer_adam_epsilon = 1e-9
    hparams.learning_rate_decay_scheme = "noam"
    hparams.learning_rate = 0.1
    hparams.learning_rate_warmup_steps = 4000
    hparams.initializer_gain = 1.0
    hparams.num_hidden_layers = 6
    hparams.initializer = "uniform_unit_scaling"
    hparams.weight_decay = 0.0
    hparams.optimizer_adam_beta1 = 0.9
    hparams.optimizer_adam_beta2 = 0.98
    hparams.num_sampled_classes = 0
    hparams.label_smoothing = 0.1
    hparams.shared_embedding_and_softmax_weights = int(True)

    hparams.layer_preprocess_sequence = ""
    hparams.layer_postprocess_sequence = "dan"
    # dropout rate to use during layer_preprocess and layer_postprocess
    hparams.layer_prepostprocess_dropout = 0.1

    hparams.add_hparam("filter_size", 2048)  # Add new ones like this.
    # attention-related flags
    hparams.add_hparam("num_heads", 8)
    hparams.add_hparam("attention_key_channels", 0)
    hparams.add_hparam("attention_value_channels", 0)
    hparams.add_hparam("ffn_layer", "conv_hidden_relu")
    hparams.add_hparam("parameter_attention_key_channels", 0)
    hparams.add_hparam("parameter_attention_value_channels", 0)
    # All hyperparameters ending in "dropout" are automatically set to 0.0
    # when not in training mode.
    hparams.add_hparam("attention_dropout", 0.0)
    hparams.add_hparam("relu_dropout", 0.0)
    hparams.add_hparam("pos", "timing")  # timing, none
    hparams.add_hparam("nbr_decoder_problems", 1)
    hparams.add_hparam("proximity_bias", int(False))
    return hparams


def transformer_tiny():
    hparams = transformer_base()
    hparams.num_hidden_layers = 2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams


if __name__ == '__main__':
    # this test case looks somewhat right....

    import numpy as np

    batch_size = 3
    input_length = 5
    target_length = 7
    vocab_size = 9

    inputs = -1 + np.random.random_integers(
        vocab_size, size=(batch_size, input_length, 1, 1))  #

    hparams = transformer_tiny()

    with tf.Graph().as_default(), tf.Session() as session:
        features = {
            "inputs": tf.constant(inputs, dtype=tf.int32)
        }
        encoder = TransformerEncoder(hparams)
        shadred_logits = encoder.model_fn(features)

        logits = tf.concat(shadred_logits, 0)  # !????
        session.run(tf.global_variables_initializer())
        res = session.run(logits)

    # assert res.shape == (batch_size, target_length, 1, 1, vocab_size)
