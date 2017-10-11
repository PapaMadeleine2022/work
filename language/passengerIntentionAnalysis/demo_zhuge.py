# coding: utf-8
#! /usr/bin/env python

import pdb
import os
import sys
import argparse
import json
import codecs
import tensorflow as tf
import random
import numpy as np
import matrix.CRF_SlotTaggingModel_multitask as Model
from keras import preprocessing

reload(sys)
sys.setdefaultencoding( "utf-8" )

def transform(sample_path):
    with codecs.open(sample_path, 'r', 'utf-8') as fr_userutter:
        userutter_txt = fr_userutter.read().strip()
    with codecs.open('./json/word2id.json', 'r', 'utf-8') as fr_word2id:
        word2id = json.loads(fr_word2id.read())
    userutter_encode_nopad = [word2id[w] for w in userutter_txt.split(' ')]
    userutter_encode = np.zeros((46))
    for i in range(len(userutter_encode_nopad)):
        userutter_encode[i] = userutter_encode_nopad[i]
    return userutter_encode, userutter_txt


def build_nn_model():
    argparams = {
        'lr': 0.001,
        'optimizer': 'adam',
        'batch_size': 1,
        'max_gradient_norm': 5.0,
        'embedding_size': 512,
        'num_layers': 1,
        'hidden_size': 256,
        'dropout_ratio': 0.5,
        'bidirectional_rnn': True,
        'patience': 10,
        'threshold': None,
        'tag_baseline_decoder': False,
        'intent_baseline_decoder': False,
        'forward_only': True,
        'maxlen_userUtter': 46,
        'char_vocab_size': 42,
        'userTag_vocab_size': 122,
        'word_vocab_size': 898,
        'userIntent_vocab_size': 18
    }

    # with tf.variable_scope("model"):
    model = Model.SlotTaggingModel(**argparams)
    return model


def prepare_input(demo_inp_utter):
    batch_true_sequence_len_test = np.sum(np.sign(demo_inp_utter))
    batch_true_sequence_len_test = np.asarray(
        [batch_true_sequence_len_test], dtype=np.int32)

    demo_final_inp_utter = np.reshape(demo_inp_utter, (-1, 1))

    sequence_length_list = [46]
    sequence_length = np.array(sequence_length_list, dtype=np.int32)

    inp = {
        'final_inp_test': demo_final_inp_utter,
        'test_sequence_length': sequence_length,
        'batch_true_sequence_len_test': batch_true_sequence_len_test,
    }

    return inp


def get_batch_viterbi_tags(logits, trans_params, sequence_lengths, maxlen):

    viterbi_sequences = []
    for logit, sequence_length in zip(logits, sequence_lengths):
        logit = logit[:sequence_length]  # keep only the valid steps
        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
            logit, trans_params)
        viterbi_sequences.append(viterbi_seq)
        assert len(
            viterbi_seq
        ) == sequence_length, 'viterbi_seq must be equal to sequence_length'
    zeropad = preprocessing.sequence.pad_sequences(
        viterbi_sequences, maxlen, padding='post', truncating='post')
    return zeropad


def predict(inp_data, model, sess):
    with codecs.open('./json/id2userIntent.json', 'r',
                     'utf-8') as fr_id2userIntent:
        id2userIntent = json.loads(fr_id2userIntent.read())
    with codecs.open('./json/id2userTag.json', 'r', 'utf-8') as fr_id2userTag:
        id2userTag = json.loads(fr_id2userTag.read())
    logits_test, trans_params_test, batch_test_intent_probs = model.batch_predict(
        sess, inp_data['final_inp_test'], inp_data['test_sequence_length'],
        inp_data['batch_true_sequence_len_test'])
    intent = np.argmax(batch_test_intent_probs)
    intent = id2userIntent[str(intent + 1)]
    tags = get_batch_viterbi_tags(logits_test, trans_params_test,
                                  inp_data['batch_true_sequence_len_test'], 46)
    tags = [id2userTag[str(t)] for t in tags[0]]
    return intent, tags


def write2file(intent, tags, user_utter):
    with codecs.open('./json/intent.json', 'r', 'utf-8') as fr_intent:
        intent_dict = json.loads(fr_intent.read())
    with codecs.open('./json/tag.json', 'r', 'utf-8') as fr_tags:
        tags_dict = json.loads(fr_tags.read())

    intent_humanize = intent_dict[intent]
    user_utter = user_utter.split(' ')
    tags_contents_list_list = []
    for i in range(len(user_utter)):
        if tags[i].startswith('tag-B'):
            t_c = [tags[i], user_utter[i]]
            tags_contents_list_list.append(t_c)
        if tags[i].startswith('tag-I'):
            tags_contents_list_list[-1][1] += ' %s' % user_utter[i]
    tags_contents = [[tags_dict[tc[0]], tc[1]]
                     for tc in tags_contents_list_list]

    # with codecs.open('./inp_out/out.txt', 'w', 'utf-8') as fw:
    #     fw.write(u'语义理解:\n')
    #     for t_c in tags_contents:
    #         fw.write('%s : %s\n' % (t_c[0], t_c[1]))
    #     fw.write(u'意图识别:\n')
    #     fw.write(u'用户意图: ' + intent_humanize)

    print (u'语义理解：\n')
    print(u'用户意图: ' + intent_humanize+'\n')
    for t_c in tags_contents:
        print('%s : %s\n' % (t_c[0], t_c[1]))
    # print(u'意图识别:\n')
    return True


def core_en(sample_path):
    userutter_encode, userutter_txt = transform(sample_path)
    with tf.Session() as sess:
        with tf.variable_scope('model', reuse=None):
            model_test = build_nn_model()
        model_test.saver.restore(sess, './model/model.ckpt-46800')

        inp_data = prepare_input(userutter_encode)
        intent, tags = predict(inp_data, model_test, sess)
        # print('===> Slot Filling:\n %s' % ' '.join(tags))
        # print('===> Intent: %s' % intent)
        write2file(intent, tags, userutter_txt)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='A Demo for Intent Detection and Slot Filling.')
    parser.add_argument(
        'sample_path',
        type=str,
        help='Path for Sample User Utterance.')
    parser.add_argument(
        '-l',
        '--lang',
        type=str,
        default='en',
        help=
        'Input \"en\" for English Demo, \"zh\" for Chinese Demo. Default is en.'
    )
    args = parser.parse_args()
    if args.sample_path is None:
        print 'Please Input Path for The Sample User Utterance.'
        return
    if not os.path.exists(args.sample_path):
        print 'No Such File or Directory.'
        return
    if args.lang == 'en':
        core_en(args.sample_path)
    if args.lang == 'zh':
        pass


if __name__ == '__main__':
    sys.exit(main())
