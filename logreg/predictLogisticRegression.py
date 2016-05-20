#!/bin/python

from __future__ import print_function

import theano.tensor as T
import theano
import os
import argparse

from LogisticRegressor import LogisticRegressor, loadParams
import numpy as np
import cPickle as pickle
import LoadData

def predict():
    parser = argparse.ArgumentParser(prog='Logistic Regression', conflict_handler='resolve',description = '''\
        This scripts predicts the classes according to a previously saved model of the provided dataset and saves it to the given output folder
        ''')
    parser.add_argument('-o', '--output', type=str, default="out", required=False, help='Path To The Output Folder')

    requiredNamed = parser.add_argument_group('Required Arguments')
    requiredNamed.add_argument('-m', '--model', type=str, required=True, help='The Previously Trained Model')
    requiredNamed.add_argument('-d', '--dataset', type=str, required=False, help='Path To The Dataset [MNIST]')

    parsed = parser.parse_args()
    if not os.path.exists(parsed.output):
        os.makedirs(parsed.output)

    params = loadParams(parsed.model)

    (train_images, train_labels), (validation_images, validation_labels), \
                (test_images, test_labels) = LoadData.loadMNIST(parsed.dataset)

    regressor = LogisticRegressor(input=test_images,labels=None, weights=params[0], bias=params[1])

    predict = theano.function(
        inputs=[],
        outputs=regressor.predictions
    )

    predictions = predict()
    hits = (predictions == test_labels.eval()).sum()
    accuracy = float(hits) / len(predictions)
    print('Num Predictions:\t%d' %(len(predictions)))
    print('Num Hits:\t\t%d' %(hits))
    print('Accuracy:\t\t%f' %(accuracy))
    out=''
    for idx in range(len(predictions)):
        out = out + str(predictions[idx]) + '\t'
        if (idx % 10) == 0:
            out = out[:-1] + '\n'
    with open(parsed.output + '/predictions.txt', 'w') as outf:
        outf.write(out)
    outf.close()
     
    
if __name__ == '__main__':
    predict()