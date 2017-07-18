#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import numpy
import cv2
import numpy as np
import cPickle as pickle
import time
import os
import codecs
__all__ = (
    'DIGITS'
)
OUTPUT_SHAPE = (48, 1193)

#DIGITS = "~!%'()+,-.\/0123456789:ABCDEFGIJKLMNOPRSTUVWYabcdefghiklmnoprstuvwxz-V،د‘“ ؤب,گ0ذصط3وLِbT2dh9ٰٴxAڈlژ؛؟أGاpث4/س7ًtCهKیُS\"۔WOcgk…ٓosw(ﷺجڑ.آئکتخز6غEشہقنضDNR8ظ:fnrvzپچB’”لء%)ْFحر5عںھف!JمIM#ّےUYَae'Pimة1uٹ+" #url
"""DIGITS = "~!%'()+,-.\/0123456789:ABCDEFGIJKLMNOPRSTUVWYabcdefghiklmnoprstuvwxz-V،د‘“ ؤب,گ0ذصط3وLِbT2dh9ٰٴxAڈlژ؛؟أGاpث4/س7ًtCهKیُS\"۔WOcgk…ٓosw(ﷺجڑ.آئکتخز6غEشہقنضDNR8ظ:fnrvzپچB’”لء%)ْFحر5عںھف!JمIM#ّےUYَae'Pimة1uٹ+".decode('utf-8')
DIGITS = DIGITS.split()
DIGITS = set(DIGITS)
DIGITS = ''.join(DIGITS)"""
DIGITS = "~!%'()+,-.\/0123456789:ABCDEFGIJKLMNOPRSTUVWYabcdefghiklmnoprstuvwxz-V،د‘“ ؤب,گ0ذصط3وLِbT2dh9ٰٴxAڈlژ؛؟أGاpث4/س7ًtCهKیُS\"۔WOcgk…ٓosw(ﷺجڑ.آئکتخز6غEشہقنضDNR8ظ:fnrvzپچB’”لء%)ْFحر5عںھف!JمIM#ّےUYَae'Pimة1uٹ+".decode('utf-8')




LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000

# parameters for bdlstm ctc
BATCH_SIZE = 100
BATCHES = 230
#100000


TRAIN_SIZE = BATCH_SIZE * BATCHES

MOMENTUM = 0.9
REPORT_STEPS = 1000

# Hyper-parameters
num_epochs = 100000
num_hidden = 256
num_layers = 1

num_classes = len(DIGITS)  + 1  # characters + ctc blank
print num_classes


data_set = {}
label_dictionary = {}

def get_labels(names):
    #print names

    for x in names:
        #x1 = os.path.basename(x)
        f = codecs.open( '0.gt.txt', 'r','utf-8')
        label_dictionary[x] = f.readline().strip('\n')

        #label_dictionary[x]=label_dictionary[x][::-1]
        f.close()


def load_data_set(dirname):
    with open(dirname) as f:
        image_names = f.readlines()
    fname_list = [x.strip() for x in image_names]
    result = dict()
    labels_list = []
    #get list of paths without extension
    for x in fname_list:
        labels_list.append((os.path.splitext(x)[0]))

    #load ground truths to label array
    get_labels(labels_list)


    for fname in sorted(fname_list):


	#print fname
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        #get corresponding label
        label_key = os.path.splitext(fname)[0]
        code = label_dictionary.get(label_key)
        result[label_key] = (im, code)

    data_set[dirname] = result


def read_data_for_lstm_ctc(dirname, start_index=None, end_index=None):
    start = time.time()
    fname_list = []

    if not data_set.has_key(dirname):
        load_data_set(dirname)

    with open(dirname) as f:
        image_names = f.readlines()
        image_names = [x.strip() for x in image_names]

    if start_index is None:
        fname_list = image_names


    else:
        for i in range(start_index, end_index):
            fname_list.append(image_names[i])

    start = time.time()
    dir_data_set = data_set.get(dirname)

    with open('testwidths.pickle', 'r') as handle:
        widths_dict = pickle.load(handle)

    for fname in sorted(fname_list):
        d = os.path.splitext(fname)[0]
        im, code = dir_data_set[d]

        #width = widths_dict[int(os.path.splitext(d)[0])]
        width=1193


        yield width, numpy.asarray(d), im, numpy.asarray([DIGITS.find(x)  for x in list(code)])


def unzip(b):
    ws, ns, xs, ys = zip(*b)
    ws = numpy.array(ws)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    ns = numpy.array(ns)
    return ws, ns, xs, ys
