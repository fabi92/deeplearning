#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import timeit
import theano.tensor as T
import theano
import argparse
from LogisticRegressor import LogisticRegressor
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle

import LoadData
from Plotter import plotError

def trainRegressor():
    parser = argparse.ArgumentParser(prog='Logistic Regression', conflict_handler='resolve',description = '''\
        This script should enable the user to train his Logistic Regression Model according to the input parameters
        ''')
    parser.add_argument('-l', '--learningrate', type=float, default=0.01, required=False, help='The Learning Rate')
    parser.add_argument('-b', '--batchsize', type=int, default=20, required=False, help='The Batch Size')
    parser.add_argument('-o', '--output', type=str, default="out", required=False, help='Path To The Output Folder')
    parser.add_argument('-e', '--epochs', type=int, default=200, required=False, help='Maximum Number Of Epochs')
    parser.add_argument('-p', '--plot', type=bool, default=False, required=False, help='Set To True In Order To Plot Error Curves')


    requiredNamed = parser.add_argument_group('Required Arguments')
    requiredNamed.add_argument('-d', '--dataset', type=str, required=True, help='Path To The Training Set')
   
    parsed = parser.parse_args()

    if not os.path.exists(parsed.output):
        os.makedirs(parsed.output)

    (train_images, train_labels), (validation_images, validation_labels), \
         (test_images, test_labels) = LoadData.loadMNIST(parsed.dataset)

    number_train_images_batches = train_images.get_value(borrow=True).shape[0] // parsed.batchsize
    number_validation_images_batches = validation_images.get_value(borrow=True).shape[0] // parsed.batchsize
    number_test_images_batches = test_images.get_value(borrow=True).shape[0] // parsed.batchsize

    index = T.lscalar() 
    imageData = T.matrix('imageData')
    imageLabels = T.ivector('imageLabels')

    regressor = LogisticRegressor(input=imageData, labels=imageLabels, n_in=28 * 28, n_out= 10)

    trainBatchGivenIndex = theano.function(
        inputs=[index],
        outputs= [regressor.cost],
        updates= [(regressor.weights, regressor.weights - parsed.learningrate * T.grad(cost=regressor.cost, wrt=regressor.weights)),
               (regressor.bias, regressor.bias - parsed.learningrate * T.grad(cost=regressor.cost, wrt=regressor.bias))],
        givens={
            imageData: train_images[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            imageLabels: train_labels[index * parsed.batchsize: (index + 1) * parsed.batchsize]
        }
    )

    trainAccuracyGivenIndex = theano.function(
        inputs=[index],
        outputs=regressor.missclassified,
        givens={
            imageData: train_images[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            imageLabels: train_labels[index * parsed.batchsize: (index + 1) * parsed.batchsize]
        }
    )

    valdiationAccuracyGivenIndex = theano.function(
        inputs=[index],
        outputs=regressor.missclassified,
        givens={
            imageData: validation_images[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            imageLabels: validation_labels[index * parsed.batchsize: (index + 1) * parsed.batchsize]
        }
    )

    testAccuracyGivenIndex = theano.function(
        inputs=[index],
        outputs=regressor.missclassified,
        givens={
            imageData: test_images[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            imageLabels: test_labels[index * parsed.batchsize: (index + 1) * parsed.batchsize]
        }
    )

    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995 
    best_validation_loss = np.inf
    best_validation_epoch = 0
    best_testing_loss = np.inf
    best_testing_epoch = 0
    test_score = 0.
 
    if parsed.plot:
        trainRes = [[],[]]
        valRes = [[],[]]
        testRes = [[],[]]

    done_looping = False
    val_freq = min(number_train_images_batches, patience // 2)
    epoch = 0

    while epoch < parsed.epochs and (not done_looping):
        epoch = epoch + 1

        for minibatch_index in range(number_train_images_batches):

            minibatch_avg_cost = trainBatchGivenIndex(minibatch_index)
            idx = (epoch - 1) * number_train_images_batches + minibatch_index

            if (idx + 1) % val_freq == 0:
                validation_losses = [valdiationAccuracyGivenIndex(currentValidationBatch)
                                     for currentValidationBatch in range(number_validation_images_batches)]
                this_validation_loss = np.mean(validation_losses)
                print("Epoch %d, Batch Index: %d / %d, Accuracy On Validation Samples: %f" \
                    % (epoch, minibatch_index,  number_train_images_batches, (100 - this_validation_loss * 100)))       

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, idx * patience_increase)
                        best_validation_epoch = epoch

                    best_validation_loss = this_validation_loss
                    test_losses = [testAccuracyGivenIndex(currentTestBatch)
                                   for currentTestBatch in range(number_test_images_batches)]
                    test_score = np.mean(test_losses)
                    print("\tEpoch %d, Batch Index: %d / %d, Accuracy On Test Samples: %f" \
                        % (epoch, minibatch_index,  number_train_images_batches, (100 - test_score * 100)))
                    if test_score < best_testing_loss:
                        print('\t\tNew Best Test Score\n\t\tSaving Network')
                        best_testing_loss = test_score
                        best_testing_epoch = epoch
                        regressor.saveRegressor(parsed.output)

            if patience <= idx:
                done_looping = True
                break

        if parsed.plot:
            print("Collecting Accuracy After Epoch %d" % (epoch))
            trainRes[1].append(np.mean([trainAccuracyGivenIndex(currentTrainBatch) \
                     for currentTrainBatch in range(number_train_images_batches)]) *100)
            valRes[1].append(np.mean([valdiationAccuracyGivenIndex(currentValidationBatch) \
                     for currentValidationBatch in range(number_validation_images_batches)]) *100)
            testRes[1].append(np.mean([testAccuracyGivenIndex(currentTestBatch) \
                     for currentTestBatch in range(number_test_images_batches)]) *100)
            trainRes[0].append(epoch)
            valRes[0].append(epoch)
            testRes[0].append(epoch)
   

    print('Optimization complete with best test score of %f %%,'
             % (100 - best_testing_loss * 100.))
    if parsed.plot:
        plotError(trainRes, valRes, testRes, parsed.output, 'error.png')

    
if __name__ == '__main__':
    trainRegressor()