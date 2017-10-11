# -*- coding: utf-8 -*-
"""
Created on Sun Feb  28 11:32:21 2016

@author: Bing Liu (liubing@cmu.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
# from six.moves import zip     # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf
from matrix import modules
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import static_rnn
from matrix.utils import to_categorical
from tensorflow.contrib.keras import preprocessing

from tensorflow.python.ops import rnn_cell_impl
from matrix.modules import *

linear = rnn_cell_impl._linear


def attention_RNN(
        encoder_outputs,
        encoder_state,
        num_decoder_symbols,
        num_heads=1,
        dtype=tf.float32,
        scope=None, ):

    print('Use the attention RNN model')
    if num_heads < 1:
        raise ValueError(
            "With less than 1 heads, use a non-attention decoder.")

    with tf.variable_scope(scope or "attention_RNN"):
        output_size = encoder_outputs[0].get_shape()[1].value
        top_states = [
            tf.reshape(e, [-1, 1, output_size]) for e in encoder_outputs
        ]
        attention_states = tf.concat(top_states, 1)
        # print(attention_states.get_shape())
        if not attention_states.get_shape()[1:2].is_fully_defined():
            raise ValueError(
                "Shape[1] and [2] of attention_states must be known: %s" %
                attention_states.get_shape())
        # attention_states is just reshape encoder_outputs(max_seq length list of (batch_size, output_size) element)
        batch_size = tf.shape(top_states[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value
        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = tf.reshape(attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = tf.get_variable("AttnW_%d" % a,
                                [1, 1, attn_size,
                                 attention_vec_size])  # conv1d kernel
            hidden_features.append(
                tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(tf.get_variable("AttnV_%d" % a, [attention_vec_size]))
        # print(hidden.get_shape())
        # print(hidden_features[0].get_shape())
        # exit()
        def attention(query, extra=None):
            """Put attention masks on hidden using hidden_features and query."""
            attn_weights = []
            ds = []  # Results of attention reads will be stored here.
            for i in xrange(num_heads):
                with tf.variable_scope("Attention_%d" % i):
                    # y = linear(query, attention_vec_size, True)
                    # y = linear(query, attention_vec_size, True)
                    y = tf.reshape(query, [-1, 1, 1, attention_vec_size])
                    # z = linear(extra, attention_vec_size, True)
                    # z = tf.reshape(extra, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(v[i] * tf.tanh(hidden_features[i] + y),
                                      [2, 3])
                    # print(s.get_shape())
                    # exit()
                    a = tf.nn.softmax(s)
                    attn_weights.append(a)
                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(
                        tf.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                        [1, 2])
                    ds.append(tf.reshape(d, [-1, attn_size]))
            return attn_weights, ds

        batch_attn_size = tf.stack([batch_size, attn_size])
        attns = [
            tf.zeros(batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
        ]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])

        # loop through the encoder_outputs
        attention_encoder_outputs = list()
        sequence_attention_weights = list()
        for i in xrange(len(encoder_outputs)):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            if i == 0:
                # with tf.variable_scope("Initial_Decoder_Attention"):
                with tf.variable_scope("Initial_Decoder_Attention"):
                    initial_state = linear(encoder_state, output_size, True)
                # with tf.variable_scope("Intent_Decoder_Attention"):
                #     intent_ouput = linear(intent, output_size, True)
                # attn_weights, ds = attention(initial_state, intent_ouput)
                attn_weights, ds = attention(initial_state)
            else:
                attn_weights, ds = attention(encoder_outputs[i])
            output = tf.concat([ds[0], encoder_outputs[i]], 1)
            # NOTE: here we temporarily assume num_head = 1
            with tf.variable_scope("AttnRnnOutputProjection"):
                logit = linear(output, num_decoder_symbols, True)
            attention_encoder_outputs.append(logit)
            # NOTE: here we temporarily assume num_head = 1
            sequence_attention_weights.append(attn_weights[0])
            # NOTE: here we temporarily assume num_head = 1

    return attention_encoder_outputs, sequence_attention_weights


def _step(time, sequence_length, min_sequence_length, max_sequence_length,
          zero_logit, generate_logit):
    # Step 1: determine whether we need to call_cell or not
    empty_update = lambda: zero_logit
    logit = control_flow_ops.cond(time < max_sequence_length, generate_logit,
                                  empty_update)

    # Step 2: determine whether we need to copy through state and/or outputs
    existing_logit = lambda: logit

    def copy_through():
        # Use broadcasting select to determine which values should get
        # the previous state & zero output, and which values should get
        # a calculated state & output.
        copy_cond = (time >= sequence_length)
        return tf.where(copy_cond, zero_logit, logit)

    logit = control_flow_ops.cond(time < min_sequence_length, existing_logit,
                                  copy_through)
    logit.set_shape(zero_logit.get_shape())
    return logit


def multi_attn(encoder_outputs,
               intent,
               sequence_length,
               forward_only,
               scope=None):
    with tf.variable_scope(scope or "non-attention_RNN"):
        attention_encoder_outputs = list()
        sequence_attention_weights = list()

        # copy over logits once out of sequence_length
        if encoder_outputs[0].get_shape().ndims != 1:
            (fixed_batch_size,
             output_size) = encoder_outputs[0].get_shape().with_rank(2)
        else:
            fixed_batch_size = encoder_outputs[
                0].get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = tf.shape(encoder_outputs[0])[0]

        encoder_outputs_tensor = tf.transpose(
            tf.stack(encoder_outputs), [1, 0, 2])

        rnn_hidden = encoder_outputs_tensor

        float_mask = tf.sequence_mask(sequence_length, len(encoder_outputs))
        float_mask = tf.cast(float_mask, tf.float32)

        for i in range(1):
            with tf.variable_scope("query"):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    rnn_hidden = multihead_attention(
                        queries=rnn_hidden,
                        keys=rnn_hidden,
                        sequence_mask=float_mask,
                        num_units=output_size,
                        num_heads=4,
                        dropout_rate=0.1,
                        is_training=not forward_only,
                        causality=False)
                    ### Feed Forward
                    rnn_hidden = feedforward(
                        rnn_hidden,
                        num_units=[4 * int(output_size), output_size],
                        kernel_size=1)

        label = tf.tile(
            tf.expand_dims(intent, 1), [1, len(encoder_outputs), 1])

        query = tf.concat([rnn_hidden, label], -1)

        h = encoder_outputs_tensor

        h_ = tf.layers.conv1d(query, h.get_shape()[-1].value, 1)
        # g = tf.layers.dense(query, h.get_shape()[-1].value, activation=tf.sigmoid)
        # h = (1-g)*h_  + g * h
        h = h_ * h
        # h = tf.concat([encoder_outputs_tensor1, label], -1)
        # logits = tf.layers.dense(h, num_decoder_symbols)
        # attention_encoder_outputs = tf.unstack(tf.transpose(logits, [1, 0, 2]))
        return h


def generate_sequence_output(
        encoder_outputs,
        encoder_state,
        targets,
        tag_weights,
        num_decoder_symbols,
        # intent,
        tag_baseline_decoder,
        name=None, ):
    all_inputs = encoder_outputs + targets
    with tf.name_scope(name, "model_with_buckets", all_inputs):
        with tf.variable_scope("decoder_sequence_output", reuse=None):
            # dim = encoder_outputs[0].get_shape()[-1].value
            # intent_output = linear(intent, dim, 0)
            # print('shape of intent',intent.get_shape())
            if tag_baseline_decoder:
                print('tagging: Use the baseline model')
                encoder_outputs_tensor = tf.stack(encoder_outputs)
                logits = tf.layers.dense(encoder_outputs_tensor,
                                         num_decoder_symbols)
                logits = tf.unstack(logits)
            # else:
            #     print('tagging: Use the joint intent model')
            #     total_s = []
            #     for one_output in encoder_outputs:
            #         s = tf.reduce_sum(one_output*intent, -1)
            #         total_s.append(s)
            #     total_s_tensor = tf.transpose(tf.stack(total_s))
            #     # print('total_s_tensor shape:', total_s_tensor.get_shape())
            #     soft_max_s = tf.nn.softmax(total_s_tensor)
            #     # print('soft_max_s shape:', soft_max_s.get_shape())
            #     soft_max_s_list = tf.unstack(tf.transpose(soft_max_s))
            #     weighted_intent_output = []
            #     for one_soft_max_s in soft_max_s_list:
            #         trans_intent_output = tf.transpose(intent)
            #         token_output = one_soft_max_s*trans_intent_output
            #         # print('shape of token',token_output.get_shape())
            #         weighted_intent_output.append(tf.transpose(token_output))
            #     weighted_intent_output = tf.stack(weighted_intent_output)
            #     # # add_encoder_outputs_tensor = tf.concat([tf.stack(encoder_outputs), weighted_intent_output], -1)
            #     add_encoder_outputs_tensor = tf.add(tf.stack(encoder_outputs), weighted_intent_output)
            #     # add_encoder_outputs = tf.unstack(add_encoder_outputs_tensor)
            #     # logits, _ = attention_RNN(add_encoder_outputs,
            #     #                           encoder_state,
            #     #                           num_decoder_symbols
            #     #                           )
            #     logits = tf.layers.dense(add_encoder_outputs_tensor, num_decoder_symbols)
            #     logits = tf.unstack(logits)

        assert len(logits) == len(targets)

        crossent = tf.nn.softmax_cross_entropy_with_logits(
            labels=targets, logits=logits)
        crossent = crossent * tag_weights
        batch_size = tf.shape(targets)[1]
        loss = tf.reduce_sum(crossent) / tf.cast(batch_size, tf.float32)

    return logits, loss


def JointCRFtagger(encoder_outputs,
                   intent,
                   sequence_length,
                   num_decoder_symbols,
                   tags_pad,
                   sentence_length,
                   forward_only=False):
    print('tag: Using CRFtagger')
    with tf.variable_scope("decoder_sequence_output", reuse=None):
        # encoder_outputs_tensor = tf.transpose(tf.stack(encoder_outputs),[1, 0, 2])
        # total_s = []
        # for one_output in encoder_outputs:
        #     s = tf.reduce_sum(one_output * intent, -1)
        #     total_s.append(s)
        # total_s_tensor = tf.transpose(tf.stack(total_s))
        # # print('total_s_tensor shape:', total_s_tensor.get_shape())
        # soft_max_s = tf.nn.softmax(total_s_tensor)
        # # print('soft_max_s shape:', soft_max_s.get_shape())
        # soft_max_s_list = tf.unstack(tf.transpose(soft_max_s))
        # weighted_intent_output = []
        # for one_soft_max_s in soft_max_s_list:
        #     trans_intent_output = tf.transpose(intent)
        #     token_output = one_soft_max_s * trans_intent_output
        #     # print('shape of token',token_output.get_shape())
        #     weighted_intent_output.append(tf.transpose(token_output))
        # weighted_intent_output = tf.stack(weighted_intent_output)
        # weighted_i =tf.transpose(weighted_intent_output, [1, 0, 2])
        # # i = tf.tile(tf.expand_dims(intent, 1), [1, len(encoder_outputs), 1])
        # # h = tf.transpose(tf.stack(encoder_outputs),[1, 0, 2])
        # # size = h.get_shape()[-1].value
        # # o = tf.layers.conv1d(tf.concat([h, i], -1), size, 1)
        # # weighted_o = tf.layers.conv1d(tf.concat([h, weighted_i], -1), size, 1)
        # # add_o = tf.add(h, weighted_i)
        # # # i = tf.transpose(weighted_intent_output,[1, 0, 2])
        # # bias_start = 0.0
        # # bias = tf.get_variable("Out_Bias",[size],initializer=tf.constant_initializer(bias_start))
        # # transform_gate = tf.sigmoid(tf.layers.conv1d(h, size, 1) + bias)
        # # carry_gate = 1. - transform_gate
        # # encoder_outputs_tensor = transform_gate * weighted_o + carry_gate * h
        # encoder_outputs_tensor = tf.transpose(tf.add(tf.stack(encoder_outputs), weighted_intent_output),[1, 0, 2])
        encoder_outputs_tensor = multi_attn(
            encoder_outputs,
            intent,
            sequence_length,
            forward_only=forward_only)
        logits = tf.layers.dense(encoder_outputs_tensor, num_decoder_symbols)
        labels = tf.transpose(tf.stack(tags_pad))
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            logits, labels, sentence_length)
        loss = tf.reduce_mean(-log_likelihood)

    return loss, logits, trans_params


def CRFtagger(encoder_outputs, num_decoder_symbols, tags_pad, sentence_length):
    print('tag: Using CRFtagger')
    with tf.variable_scope("decoder_sequence_output", reuse=None):
        encoder_outputs_tensor = tf.transpose(
            tf.stack(encoder_outputs), [1, 0, 2])
        logits = tf.layers.dense(encoder_outputs_tensor, num_decoder_symbols)
        labels = tf.transpose(tf.stack(tags_pad))
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            logits, labels, sentence_length)
        loss = tf.reduce_mean(-log_likelihood)

    return loss, logits, trans_params
