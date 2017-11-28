#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import codecs
import argparse
from matrix.sequences_test import solve

def write2file(result):
    predict, right, is_right = result
    with codecs.open('./results/result.txt', 'w', 'utf-8') as fw:
        fw.write(u'正确答案为：%s\n' % right)
        print u'正确答案为：%s' % right
        fw.write(u'预测答案为：%s\n' % predict)
        print u'预测答案为：%s' % predict
        if is_right:
            fw.write(u'预测正确！')
            print u'预测正确！'
        else:
            fw.write(u'预测错误！')
            print u'预测错误！'

def main(argv=None):
    parser = argparse.ArgumentParser(
        description='A Demo for Multi-Choice Question Answering in Exams.')
    parser.add_argument(
        '-s',
        '--sample_path',
        type=str,
        help='Path for Sample Question.')
    args = parser.parse_args()
    if args.sample_path is None:
        print 'Please Input Path for The Sample User Utterance.'
        return
    result = solve(args.sample_path)
    write2file(result)


if __name__ == '__main__':
    sys.exit(main())
