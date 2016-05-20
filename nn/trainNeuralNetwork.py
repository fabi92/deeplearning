#!/usr/bin/python

from __future__ import print_function

import os
import sys
import timeit

import numpy as np

from NeuralNetwork import FFNN
import theano
import theano.tensor as T
import argparse
import cPickle as pickle

import LoadData
from Plotter import plotError

activation_functions = {
        "tanh": T.tanh,
        "relu": T.nnet.relu,
        "sigmoid": T.nnet.sigmoid,
   }

def train_network():
    ## parses the provided parameters according to the command line input
    parser = argparse.ArgumentParser(prog='Feed Forward Neural Network', conflict_handler='resolve',description = '''\
        This script should enable the user to train his Neural Network according to the input parameters '''\
        '''\nNormally, the network only consists out of a input and a output layer. However the possibility exists to add as many hiddenlayers as wanted
        ''')
    
    parser.add_argument('-h', '--hiddenlayers', type=int, default=1,  required=False, help='Number of Hidden Layers')
    parser.add_argument('-u', '--unitshidden', type=int, default=300,  required=False, help='Number of Hidden Units')
    parser.add_argument('-l', '--learningrate', type=float, default=0.01, required=False, help='The Learning Rate')
    parser.add_argument('-m', '--momentum', type=float, default=0.01, required=False, help='The Momentum Rate')
    parser.add_argument('-w', '--weight_decay', type=float, default=5e-4, required=False, help='The Weight Decay Rate')
    parser.add_argument('-b', '--batchsize', type=int, default=20, required=False, help='Batch Size For Training')
    parser.add_argument('-h', '--hiddenunits', type=int, default=300,  required=False, help='Number of Hidden Units')
    parser.add_argument('-o', '--output', type=str, default="out", required=False, help='Path To The Output Folder')
    parser.add_argument('-1', '--l1reg', type=float, default=0.001, required=False, help='Value For L1 Regularisaion')
    parser.add_argument('-2', '--l2reg', type=float, default=0.0001, required=False, help='Value For L2 Regularisaion')
    parser.add_argument('-e', '--epochs', type=int, default=500, required=False, help='Number Of Epochs')
    parser.add_argument('-a', '--activation', type=str, default="tanh", choices=["tanh", "sigmoid", "relu"]
        , required=False, help='Activation Functions For The Perceptrons [tanh/sigmoid/relu]')  
    parser.add_argument('-d', '--dropout', type=bool, default=False, required=False, help='Set To True In Order To Activate Dropout')
    parser.add_argument('-p', '--plot', type=bool, default=False, required=False, help='Set To True In Order To Plot Error Curves')


    requiredNamed = parser.add_argument_group('required Arguments')
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
    isTrain = T.iscalar('is_train')

    rng = np.random.RandomState(1234)##numpy random range generator

    assert parsed.epochs > 0 and parsed.hiddenlayers > 0
    network = FFNN(
        input=imageData,
        rng=rng,
        n_hiddenlayer=parsed.hiddenlayers,
        n_in=28*28, ##image 28x28
        n_hidden=parsed.hiddenunits,
        n_out=10,  ##10 classes [0,9]
        activation=activation_functions[parsed.activation],
        isTrain=isTrain,
        dropout=parsed.dropout
    )

    cost = (
        network.cost(imageLabels) + parsed.l1reg * network.l1Regularisation() + parsed.l2reg * network.l2Regularisation()
    )

    assert parsed.momentum >= 0. and parsed.momentum < 1.
    updates = []
    for param in  network.params:
        last_update = theano.shared(param.get_value()*np.cast[theano.config.floatX](0.))
        delta = -parsed.learningrate * T.grad(cost, param) + parsed.momentum * last_update - \
                    parsed.learningrate * parsed.weight_decay * param
        updates.append((param, param + delta))
        updates.append((last_update, delta))


    trainBatchGivenIndex = theano.function(
        inputs=[index],
        outputs= [cost],
        updates= updates,
        givens={
            imageData: train_images[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            imageLabels: train_labels[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            isTrain: np.cast['int32'](1)
        }
    )

    trainAccuracyGivenIndex = theano.function(
        inputs=[index],
        outputs=network.missclassified(imageLabels),
        givens={
            imageData: train_images[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            imageLabels: train_labels[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            isTrain: np.cast['int32'](0)
        }
    )

    valdiationAccuracyGivenIndex = theano.function(
        inputs=[index],
        outputs=network.missclassified(imageLabels),
        givens={
            imageData: validation_images[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            imageLabels: validation_labels[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            isTrain: np.cast['int32'](0)
        }
    )

    testAccuracyGivenIndex = theano.function(
        inputs=[index],
        outputs=network.missclassified(imageLabels),
        givens={
            imageData: test_images[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            imageLabels: test_labels[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            isTrain: np.cast['int32'](0)
        }
    )

    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.9875 
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

    while (epoch < parsed.epochs) and (not done_looping):
        epoch = epoch + 1

        for minibatch_index in range(number_train_images_batches):

            minibatch_avg_cost = trainBatchGivenIndex(minibatch_index)
            idx = (epoch - 1) * number_train_images_batches + minibatch_index

            if (idx + 1) % val_freq == 0:
                validation_losses = [valdiationAccuracyGivenIndex(currentValidationBatch)
                                     for currentValidationBatch in range(number_validation_images_batches)]
                this_validation_loss = np.mean(validation_losses)
                print("Epoch %d, Batch Index: %d / %d, Loss wrt Validation Samples: %f" \
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
                        print('\t\tNew Best Test Result\n\t\tSaving Network')
                        best_testing_loss = test_score
                        best_testing_epoch = epoch
                        network.save(path=parsed.output, name='nn_' + str(parsed.activation))

            if patience <= idx:
                done_looping = True
                break
        if parsed.plot:
            print("Collecting Losses After Epoch %d" % (epoch))
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
        plotError(trainRes, valRes, testRes, parsed.output, 'error_' + parsed.activation + '.png')

    
if __name__ == '__main__':
    train_network()