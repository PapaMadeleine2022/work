''' Natural language understanding model based on multi-task learning.
    This model is trained on two tasks: slot tagging and user intent prediction.

    Inputs: user utterance, e.g. BOS w1 w2 ... EOS
    Outputs: slot tags and user intents, e.g. O O B-moviename ... O\tinform+moviename

    Author      : Xuesong Yang
    Email       : xyang45@illinois.edu
    Created Date: Dec. 31, 2016
'''
import os
import time
import sys
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve

from matrix.DataSetCSVslotTagging import DataSetCSVslotTagging
from matrix.utils import print_params, intent_accuracy, crf_eval_slotTagging, crf_getNLUframeAccuracy, crf_getNLUpred, eval_slotTagging, eval_intentPredict, writeTxt, getNLUpred, getActPred, getTagPred, checkExistence, getNLUframeAccuracy, eval_actPred, to_categorical
from matrix import seq_classification
from matrix import seq_labeling
from matrix import modules

from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import static_rnn
from tensorflow.contrib.rnn import static_bidirectional_rnn
from tensorflow.python.ops import rnn_cell_impl
from keras import preprocessing
linear = rnn_cell_impl._linear
np.random.seed(1983)

_MAX_CHAR_L = 20
_MAX_INTENT_L = 3


def writeUtterTagIntentTxt(utter_txt, tag_txt, intent_txt, target_fname):
    with open(target_fname, 'wb') as f:
        for (utter, tag, intent) in zip(utter_txt, tag_txt, intent_txt):
            tag_new = [token.replace('tag-', '', 1) for token in tag.split()]
            intent_new = [
                token.replace('intent-', '', 1) for token in intent.split(';')
            ]
            new_line = '{}\t{}\t{}'.format(utter, ' '.join(tag_new),
                                           ';'.join(intent_new))
            f.write('{}\n'.format(new_line))


def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    output = input_
    for idx in xrange(layer_size):
        output = f(linear(output, size, 0, scope='output_lin_%d' % idx))
        transform_gate = tf.sigmoid(
            linear(input_, size, 0, scope='transform_lin_%d' % idx) + bias)
        carry_gate = 1. - transform_gate
        output = transform_gate * output + carry_gate * input_

    return output


class SlotTaggingModel(object):
    def __init__(self, **argparams):
        self.learning_rate = argparams['lr']
        self.opt_method = argparams['optimizer']
        self.batch_size = argparams['batch_size']
        self.max_gradient_norm = argparams['max_gradient_norm']
        self.embedding_size = argparams['embedding_size']
        self.num_layers = argparams['num_layers']
        self.hidden_size = argparams['hidden_size']
        self.dropout = argparams['dropout_ratio']
        self.bidirectional_rnn = argparams['bidirectional_rnn']
        self.patience = argparams['patience']
        self.threshold = argparams['threshold']
        self.tag_baseline_decoder = argparams['tag_baseline_decoder']
        self.intent_baseline_decoder = argparams['intent_baseline_decoder']
        self.forward_only = argparams['forward_only']

        self.maxlen_userUtter = argparams['maxlen_userUtter']
        self.char_vocab_size = argparams['char_vocab_size']
        self.userTag_vocab_size = argparams['userTag_vocab_size']
        self.word_vocab_size = argparams['word_vocab_size']
        self.userIntent_vocab_size = argparams['userIntent_vocab_size']

        self.global_step = tf.Variable(0, trainable=False)

        def create_cell():
            single_cell = lambda: BasicLSTMCell(self.hidden_size)
            cell = MultiRNNCell(
                [single_cell() for _ in range(self.num_layers)])
            if not self.forward_only:
                cell = DropoutWrapper(
                    cell,
                    input_keep_prob=self.dropout,
                    output_keep_prob=self.dropout)
            return cell

        self.cell_fw = create_cell()
        self.cell_bw = create_cell()

        self.build_inputs()
        encoder_inputs = self.generate_word_encoder_inputs()
        base_rnn_output = self.generate_rnn_output(encoder_inputs)
        encoder_outputs, encoder_state, attention_states = base_rnn_output

        seq_intent_outputs = seq_classification.generate_single_output(
            encoder_outputs,
            encoder_state,
            self.sequence_length,
            self.labels,
            self.userIntent_vocab_size,
            self.hidden_size,
            self.dropout,
            forward_only=self.forward_only,
            intent_baseline_decoder=self.intent_baseline_decoder)
        self.classification_output, self.classification_loss = seq_intent_outputs

        seq_labeling_outputs = seq_labeling.CRFtagger(
            encoder_outputs,
            self.userTag_vocab_size,
            self.tags_pad,
            self.length, )
        self.tagging_loss, self.logits, self.trans_params = seq_labeling_outputs

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not self.forward_only:
            if self.opt_method == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif self.opt_method == 'adagrad':
                opt = tf.train.AdagradOptimizer(
                    learning_rate=self.learning_rate)
            elif self.opt_method == 'sgd':
                opt = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate)
            elif self.opt_method == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(
                    learning_rate=self.learning_rate)
            else:
                raise NotImplementedError(
                    "Unknown method {}".format(self.opt_method))
        # backpropagate the intent and tagging loss, one may further adjust
        # the weights for the two costs.
            gradients = tf.gradients(
                [self.classification_loss, self.tagging_loss], params)

            clipped_gradients, norm = tf.clip_by_global_norm(
                gradients, self.max_gradient_norm)
            self.gradient_norm = norm
            self.update = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

    def build_inputs(self):

        self.char_inputs = []
        self.encoder_inputs = []
        self.tags = []
        self.tags_pad = []
        self.tag_weights = []
        self.labels = []
        self.un1hot_labels = []
        self.sequence_length = tf.placeholder(
            tf.int32, [None], name="sequence_length")
        self.length = tf.placeholder(tf.int32, [None], name="length")
        self.intent_length = tf.placeholder(
            tf.int32, [None], name="intent_length")
        self.char_sequence_length = tf.placeholder(
            tf.int32, [None], name="char_sequence_length")

        for i in xrange(self.maxlen_userUtter):
            self.char_inputs.append(
                tf.placeholder(
                    tf.int32,
                    shape=[None, _MAX_CHAR_L],
                    name="char_inputs{0}".format(i)))

        for i in xrange(self.maxlen_userUtter):
            self.encoder_inputs.append(
                tf.placeholder(
                    tf.int32, shape=[None], name="encoder{0}".format(i)))
        for i in xrange(self.maxlen_userUtter):
            self.tags.append(
                tf.placeholder(
                    tf.int32,
                    shape=[None, self.userTag_vocab_size],
                    name="tag{0}".format(i)))
            self.tags_pad.append(
                tf.placeholder(
                    tf.int32, shape=[None], name="tag_pad{0}".format(i)))
            self.tag_weights.append(
                tf.placeholder(
                    tf.float32, shape=[None], name="weight{0}".format(i)))
        self.labels.append(
            tf.placeholder(
                tf.float32,
                shape=[None, self.userIntent_vocab_size],
                name="label"))
        self.un1hot_labels.append(
            tf.placeholder(
                tf.float32, shape=[None, _MAX_INTENT_L], name="un1hot_label"))

    def generate_rnn_output(self, encoder_emb_inputs):
        """
        Generate RNN state outputs with word embeddings as inputs
        """
        with tf.variable_scope("generate_seq_output"):
            if self.bidirectional_rnn:
                print('encoder: Use Bi-LSTM')
                rnn_outputs = static_bidirectional_rnn(
                    self.cell_fw,
                    self.cell_bw,
                    encoder_emb_inputs,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32)
                encoder_outputs, encoder_state_fw, encoder_state_bw = rnn_outputs
                # with state_is_tuple = True, if num_layers > 1,
                # here we simply use the state from last layer as the encoder state
                state_fw = encoder_state_fw[-1]
                state_bw = encoder_state_bw[-1]
                encoder_state = tf.concat(
                    [tf.concat(state_fw, 1),
                     tf.concat(state_bw, 1)], 1)
                top_states = [tf.reshape(e, [-1, 1, self.cell_fw.output_size \
                                             + self.cell_bw.output_size])
                              for e in encoder_outputs]
                attention_states = tf.concat(top_states, 1)
            else:
                print('encoder: Use Single-LSTM')
                rnn_outputs = static_rnn(
                    self.cell_fw,
                    encoder_emb_inputs,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32)
                encoder_outputs, encoder_state = rnn_outputs
                # with state_is_tuple = True, if num_layers > 1,
                # here we use the state from last layer as the encoder state
                state = encoder_state[-1]
                encoder_state = tf.concat(state, 1)
                top_states = [
                    tf.reshape(e, [-1, 1, self.cell_fw.output_size])
                    for e in encoder_outputs
                ]
                attention_states = tf.concat(top_states, 1)
            return encoder_outputs, encoder_state, attention_states

    def generate_word_encoder_inputs(self):
        """
        Generate RNN state outputs with word embeddings and char embedding as inputs
        """
        print('Using word embedding')
        embedding = tf.get_variable(
            "embedding", [self.word_vocab_size, self.embedding_size])
        embedding = tf.concat(
            (tf.zeros(shape=[1, self.embedding_size]), embedding[1:, :]), 0)

        encoder_emb_inputs = [tf.nn.embedding_lookup(embedding, encoder_input) \
                              for encoder_input in self.encoder_inputs]
        return encoder_emb_inputs

    def generate_char_embed(self):
        """
        Generate RNN state outputs with word embeddings and char embedding as inputs
        """
        print('Using char embedding concat with word embedding')
        char_embedding = tf.get_variable(
            "char_embedding", [self.char_vocab_size, self.embedding_size])
        char_embedding = tf.concat(
            (tf.zeros(shape=[1, self.embedding_size]), char_embedding[1:, :]),
            0)
        char_emb_inputs = [
            tf.nn.embedding_lookup(char_embedding, char_input)
            for char_input in self.char_inputs
        ]

        embedding = tf.get_variable(
            "embedding", [self.word_vocab_size, self.embedding_size / 2])
        embedding = tf.concat(
            (tf.zeros(shape=[1, self.embedding_size / 2]), embedding[1:, :]),
            0)

        word_emb = [tf.nn.embedding_lookup(embedding, encoder_input) \
                    for encoder_input in self.encoder_inputs]

        encoder_emb_inputs = [
            tf.concat([a, b], -1) for a, b in zip(cnn_outputs, word_emb)
        ]

        return encoder_emb_inputs

    def batch_fit(self, session, char_inputs, inputs, tags, tags_pad,
                  tag_weights, intents, un1hot_intents, batch_sequence_length,
                  batch_true_seq_length, batch_true_intent_length,
                  char_sequence_length):
        input_feed = {}
        input_feed[self.sequence_length.name] = batch_sequence_length
        input_feed[self.length.name] = batch_true_seq_length
        input_feed[self.intent_length.name] = batch_true_intent_length
        input_feed[self.char_sequence_length.name] = char_sequence_length
        for i in xrange(self.maxlen_userUtter):
            input_feed[self.char_inputs[i].name] = char_inputs[i]
            input_feed[self.encoder_inputs[i].name] = inputs[i]
            input_feed[self.tags[i].name] = tags[i]
            input_feed[self.tags_pad[i].name] = tags_pad[i]
            input_feed[self.tag_weights[i].name] = tag_weights[i]

        input_feed[self.labels[0].name] = intents
        input_feed[self.un1hot_labels[0].name] = un1hot_intents

        output_feed = [
            self.update,  # Update Op that does SGD.
            self.gradient_norm,  # Gradient norm.
            self.logits,
            self.trans_params,
            self.tagging_loss,
            self.classification_loss,
            self.classification_output
        ]  # Loss for this batch
        outputs = session.run(output_feed, input_feed)

        return outputs[2], outputs[3], outputs[4], outputs[5], outputs[-1]

    def batch_predict(self, session, inputs, sequence_length,
                      batch_true_seq_length):

        input_feed = {}
        input_feed[self.sequence_length.name] = sequence_length
        input_feed[self.length.name] = batch_true_seq_length
        for i in xrange(self.maxlen_userUtter):
            input_feed[self.encoder_inputs[i].name] = inputs[i]

        output_feed = [
            self.logits, self.trans_params, self.classification_output
        ]

        outputs = session.run(output_feed, input_feed)
        logits = outputs[0]
        trans_params = outputs[1]

        batch_intent_probs = outputs[-1]

        return logits, trans_params, batch_intent_probs

    def get_dev_test_batches(n, batch_size):
        batches = zip(
            range(0, n, batch_size),
            range(batch_size, n + batch_size, batch_size))
        batches = [(start, end) for start, end in batches]

        return batches
