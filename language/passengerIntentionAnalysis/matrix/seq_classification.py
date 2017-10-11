# -*- coding: utf-8 -*-
"""
Created on Sun Feb  28 15:28:44 2016

@author: Bing Liu (liubing@cmu.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
# We disable pylint because we need python3 compatibility.
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import static_rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn import static_bidirectional_rnn

linear = rnn_cell_impl._linear


def attention_single_output_decoder(initial_state,
                                    add_tensor,
                                    output_size=None,
                                    num_heads=1,
                                    dtype=tf.float32,
                                    scope=None,
                                    ):
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    attention_states =  tf.transpose(add_tensor, [1, 0, 2])
    if not attention_states.get_shape()[1:2].is_fully_defined():
        raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                         % attention_states.get_shape())

    with tf.variable_scope(scope or "decoder_single_output"):
            #    print (initial_state.eval().shape)
        batch_size = tf.shape(initial_state)[0]  # Needed for reshaping.
            #    print (attention_states.get_shape())
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value

            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = tf.reshape(
                attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = tf.get_variable("AttnW_%d" % a,
                                    [1, 1, attn_size, attention_vec_size])
            hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(tf.get_variable("AttnV_%d" % a,
                                         [attention_vec_size]))

        def attention(query):
                """Put attention masks on hidden using hidden_features and query."""
                attn_weights = []
                ds = []  # Results of attention reads will be stored here.
                for i in xrange(num_heads):
                    with tf.variable_scope("Attention_%d" % i):
                        # y = linear(query, attention_vec_size, True)
                        y = linear(query, attention_vec_size, True)
                        y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
                        # Attention mask is a softmax of v^T * tanh(...).
                        s = tf.reduce_sum(
                            v[i] * tf.tanh(hidden_features[i] + y), [2, 3])

                        a = tf.nn.softmax(s)
                        attn_weights.append(a)
                        # Now calculate the attention-weighted vector d.
                        d = tf.reduce_sum(
                            tf.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                            [1, 2])
                        ds.append(tf.reshape(d, [-1, attn_size]))
                return attn_weights, ds

        batch_attn_size = tf.stack([batch_size, attn_size])
        attns = [tf.zeros(batch_attn_size, dtype=dtype)
                     for _ in xrange(num_heads)]
        for a in attns:  # Ensure the second shape of attention vectors is set.
                a.set_shape([None, attn_size])
        attn_weights, attns = attention(initial_state)

            # with variable_scope.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Out_Matrix", [attn_size, output_size])
        res = tf.matmul(attns[0], matrix)
            # NOTE: here we temporarily assume num_head = 1
        bias_start = 0.0
        bias_term = tf.get_variable("Out_Bias",
                                        [output_size],
                                        initializer=tf.constant_initializer(bias_start))
        output = res + bias_term

    # NOTE: here we temporarily assume num_head = 1
    return attn_weights[0], attns[0], output

def self_attention(encoder_outputs,
                   encoder_state,
                   output_size,
                   num_heads=1,
                   scope=None):
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")

    with tf.variable_scope(scope or "decoder_single_output"):
        batch_size = encoder_outputs.get_shape()[0].value # Needed for reshaping.
        attn_length = encoder_outputs.get_shape()[1].value
        attn_size = encoder_outputs.get_shape()[2].value
        attention_vec_size = attn_size
        # dim = attn_size + encoder_state.get_shape()[-1].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = tf.reshape(encoder_outputs, [-1, attn_length, 1, attn_size])
        attn_weights = []
        ds = []  # Results of attention reads will be stored here.
        for a in xrange(num_heads):
            k = tf.get_variable("AttnW_%d" % a,[1, 1, attn_size, attention_vec_size])
            h = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            v = tf.get_variable("AttnV_%d" % a,[attention_vec_size])
            s = tf.reduce_sum(v * tf.tanh(h), [2, 3])
            a = tf.nn.softmax(s)
            attn_weights.append(a)
            d = tf.reduce_sum(tf.reshape(a, [-1, attn_length, 1, 1]) * hidden,[1, 2])
            ds.append(tf.reshape(d, [-1, attn_size]))

        matrix = tf.get_variable("Out_Matrix", [attn_size, output_size])
        res = tf.matmul(ds[0], matrix)
        bias_start = 0.0
        bias_term = tf.get_variable("Out_Bias",[output_size],initializer=tf.constant_initializer(bias_start))
        output = res + bias_term

    return ds[0], output

def generate_single_output(encoder_outputs, encoder_state, sequence_length,
                           targets, num_classes, hidden_size, dropout_rate,
                           forward_only=False,
                           intent_baseline_decoder=False,
                           name=None):
    all_inputs = targets
    with tf.name_scope(name, "model_with_buckets", all_inputs):
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            if intent_baseline_decoder:
                print('intent: Use the baseline model')
                with tf.variable_scope("baseline_output"):
                    # decoder_outputs_tensor = encoder_state
                    # bucket_output = tf.layers.dense(encoder_state, num_classes)
                    cell = BasicLSTMCell(hidden_size)
                    decoder_outputs, final_state = static_rnn(cell, encoder_outputs, sequence_length=sequence_length, dtype=tf.float32)
                    decoder_output = decoder_outputs[-1]
                    if not forward_only:
                        decoder_output = tf.nn.dropout(decoder_output, dropout_rate)
                    bucket_output = tf.layers.dense(decoder_output, num_classes)

            else:
                print('intent: Use the attention model')
                encoder_outputs_tensor = tf.transpose(tf.stack(encoder_outputs),[1, 0, 2])#shape(batch_size,seq_len,2*hidden_size)
                decoder_output, bucket_output = self_attention(encoder_outputs_tensor,encoder_state,output_size=num_classes,num_heads=1)

            sig_bucket_output = tf.sigmoid(bucket_output)
            crossent = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets[0], logits=bucket_output)
            batch_size = tf.shape(targets[0])[0]
            loss = tf.reduce_sum(crossent) / tf.cast(batch_size, tf.float32)

    return sig_bucket_output, loss

def generate_action_output(encoder_outputs, encoder_state,
                           targets, num_classes,
                           name=None):
    all_inputs = targets
    with tf.name_scope(name, "model_action_buckets", all_inputs):
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            print('action: Use the attention model')
            encoder_outputs_tensor = tf.transpose(tf.stack(encoder_outputs),[1, 0, 2])#shape(batch_size,seq_len,2*hidden_size)
            decoder_output, bucket_output = self_attention(encoder_outputs_tensor,encoder_state,output_size=num_classes,num_heads=1,scope=tf.get_variable_scope())

            sig_bucket_output = tf.sigmoid(bucket_output)
            crossent = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets[0], logits=bucket_output)
            batch_size = tf.shape(targets[0])[0]
            loss = tf.reduce_sum(crossent) / tf.cast(batch_size, tf.float32)

    return sig_bucket_output, loss
# def generate_action_output(encoder_outputs,
#                            sequence_length,
#                            window_length,
#                            num_TagIntents,
#                            hidden_size,
#                            num_classes,
#                            targets,
#                            dropout_rate,
#                            forward_only=False,
#                            name=None):
#     with tf.name_scope(name, "model_actions", targets):
#         with tf.variable_scope(tf.get_variable_scope(), reuse=None):
#             print('action:Using LSTM+Bi-LSTM')
#             cell = BasicLSTMCell(num_TagIntents)
#             nlu_outputs, _ = static_rnn(cell, encoder_outputs, sequence_length=sequence_length,dtype=tf.float32)
#             nlu_output = nlu_outputs[-1]
#             if not forward_only:
#                 nlu_output = tf.nn.dropout(nlu_output, dropout_rate)
#             #bi-lstm on the top of nlu for action prediction
#             cell_fw = BasicLSTMCell(hidden_size)
#             cell_bw = BasicLSTMCell(hidden_size)
#             action_window_inputs = tf.unstack(nlu_output, 1)
#             rnn_outputs = static_bidirectional_rnn(cell_fw,
#                                                    cell_bw,
#                                                    action_window_inputs,
#                                                    sequence_length=window_length,
#                                                    dtype=tf.float32)
#             action_outputs, _, _ = rnn_outputs
#             action_output = action_outputs[-1]
#             if not forward_only:
#                 action_output = tf.nn.dropout(action_output, dropout_rate)
#             bucket_output = tf.layers.dense(action_output, num_classes)
#
#             sig_bucket_output = tf.sigmoid(bucket_output)
#             crossent = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets[0], logits=bucket_output)
#             batch_size = tf.shape(targets[0])[0]
#             loss = tf.reduce_sum(crossent) / tf.cast(batch_size, tf.float32)
#
#         return sig_bucket_output, loss