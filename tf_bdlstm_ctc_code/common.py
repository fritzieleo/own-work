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
from label_generate import DIGITS
__all__ = (
    'DIGITS'
)
OUTPUT_SHAPE = (48, 1142)

DIGITS = DIGITS.decode('utf-8')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#        You can change these parameters

LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
INITIAL_LEARNING_RATE = 1e-4
DECAY_STEPS = 5000

## TRAINING BATCH SIZE ##
BATCH_SIZE = 20
BATCHES = 500

## TEST BATCH SIZE ##
TEST_BATCH_SIZE = 10
TEST_BATCHES = 200

MOMENTUM = 0.9
REPORT_STEPS = 1000 #save the model after this many steps

# Hyper-parameters
num_epochs = 1000000
num_hidden = 256
num_layers = 1

transcription_ext = '.txt'   #you may need to change this if you have a different extension e.g. '.gt.txt'
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TRAIN_SIZE = BATCH_SIZE * BATCHES

num_classes = len(DIGITS)  + 1  # characters + ctc blank
print num_classes

data_set = {}
label_dictionary = {}

def get_labels(names):
    for x in names:
        f = codecs.open( x + transcription_ext, 'r','utf-8')
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

    with open('train_widths.pickle', 'r') as handle:
        widths_dict = pickle.load(handle)
    for fname in sorted(fname_list):
        d = os.path.splitext(fname)[0]
        im, code = dir_data_set[d]
        width = widths_dict[str(os.path.basename(d))]


        yield width, numpy.asarray(d), im, numpy.asarray([DIGITS.find(x)  for x in list(code)])


def unzip(b):
    ws, ns, xs, ys = zip(*b)
    ws = numpy.array(ws)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    ns = numpy.array(ns)
    return ws, ns, xs, ys
