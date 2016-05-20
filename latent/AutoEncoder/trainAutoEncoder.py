#!/usr/bin/python

from __future__ import print_function

import os
import sys
import timeit

import numpy as np

from AutoEncoder import AutoEncoder
import theano
import theano.tensor as T
from itertools import product
import argparse
import cPickle as pickle
import matplotlib.pyplot as plt
import pylab

import LoadData
from Plotter import arraysToImgs

def train_autoencoder():

    ## parses the provided parameters according to the command line input
    parser = argparse.ArgumentParser(prog='AutoEncoder', conflict_handler='resolve',description = '''\
        This script should enable the user to train his AutoEncoder according to the input parameters
        ''')
    parser.add_argument('-l', '--learningrate', type=float, default=0.025, required=False, help='The Learning Rate')
    parser.add_argument('-b', '--batchsize', type=int, default=20, required=False, help='Batch Size For Training')
    parser.add_argument('-h', '--reducedUnits', type=int, default=30,  required=False, help='Number of Reduced Layer Units')
    parser.add_argument('-o', '--output', type=str, default="out", required=False, help='Path To The Output Folder')
    parser.add_argument('-1', '--l1reg', type=float, default=0.1, required=False, help='Value For L1 Regularisaion')
    parser.add_argument('-k', '--kul_leib_penalty', type=float, default=0.04, required=False, help='Value For Kullback Leiber Divergence Penalty')
    parser.add_argument('-k', '--kul_leib_beta', type=float, default=1.0, required=False, help='Controls The Weight Of The Sparsity Penalty Term')
    parser.add_argument('-s', '--sparsity', type=str, default='l1reg', choices=['l1reg', 'kul_leib'], required=False, help='Choose Which Penalty Should Be Used')
    parser.add_argument('-e', '--epochs', type=int, default=500, required=False, help='Number Of Epochs')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, required=False, help='The Momentum Rate')


    requiredNamed = parser.add_argument_group('Required Arguments')
    requiredNamed.add_argument('-d', '--dataset', type=str, required=True, help='Path To The Training Set (MNIST)')
   
    parsed = parser.parse_args()

    if parsed.sparsity == 'kul_leib':
        assert parsed.kul_leib_penalty < 0.05
        outpath_raw = parsed.output + "/kul_leib"
    else:
        outpath_raw = parsed.output + "/l1reg"

    if not os.path.exists(outpath_raw):
        os.makedirs(outpath_raw)

    (train_images, train_labels), (validation_images, validation_labels), \
         (test_images, test_labels) = LoadData.loadMNIST(parsed.dataset)#, shuffle=True)

    number_train_images_batches = train_images.get_value(borrow=True).shape[0] // parsed.batchsize
    number_test_images_batches = test_images.get_value(borrow=True).shape[0] // parsed.batchsize
    number_validation_images_batches = validation_images.get_value(borrow=True).shape[0] // parsed.batchsize

    index = T.lscalar() 
    imageData = T.matrix('imageData')

    rng = np.random.RandomState(1234)##numpy random range generator

    autoencoder = AutoEncoder(
        input=imageData,
        rng=rng,
        n_input=28*28, ##image 28x28
        n_reduced=parsed.reducedUnits,
        sparsity_param=parsed.kul_leib_penalty,
        beta=parsed.kul_leib_beta,
        n_reconstructed=28*28
    )

    if parsed.sparsity == 'l1reg':
        cost_sparse = (
            autoencoder.cost
            + parsed.l1reg * abs(autoencoder.reducedLayer.weights).sum()
        )
    else:
        cost_sparse = (
            autoencoder.cost + autoencoder.kul_leib
        )



    updates = (
        gradient_updates_momentum(cost_sparse, autoencoder.params, parsed.learningrate, parsed.momentum)
    )


    trainBatchGivenIndex = theano.function(
        inputs=[index],
        outputs= cost_sparse,
        updates= updates,
        givens={
            imageData: train_images[index * parsed.batchsize: (index + 1) * parsed.batchsize]
        }
    )

    validateBatchGivenIndex = theano.function(
        inputs=[index],
        outputs= cost_sparse,
        givens={
            imageData: validation_images[index * parsed.batchsize: (index + 1) * parsed.batchsize]
        }
    )

    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995 
    best_validation_loss = np.inf
    best_validation_epoch = 0

    val_freq = min(number_train_images_batches, patience // 2)
    epoch = 0

    # improvement_threshold = 0.995 
    # lowest_cost = np.inf
    # best_minibatch = -1
    # best_epoch = -1
    encoder_name = None
    if parsed.sparsity == 'l1reg':
        encoder_name = 'encoder_' + str(parsed.l1reg) + '_l1'
    else:
        encoder_name = 'encoder_' + str(parsed.kul_leib_beta) + '_kul_leib'
    
    done_looping = False
    while (epoch < parsed.epochs) and not (done_looping):
        epoch = epoch + 1
        for minibatch_index in range(number_train_images_batches):
            minibatch_squared_error_loss = trainBatchGivenIndex(minibatch_index)
            idx = (epoch - 1) * number_train_images_batches + minibatch_index

            if (idx + 1) % val_freq == 0:
                validation_losses = [validateBatchGivenIndex(currentValidationBatch)
                                     for currentValidationBatch in range(number_validation_images_batches)]
                this_validation_loss = np.mean(validation_losses)
                print("Epoch %d, Batch Index: %d / %d, Accuracy On Validation Samples: %f" \
                    % (epoch, minibatch_index,  number_train_images_batches, this_validation_loss))       
                
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, idx * patience_increase)
                        best_validation_epoch = epoch

                    autoencoder.save(outpath_raw, encoder_name)
                    lowest_cost = this_validation_loss
                    best_validation_loss = this_validation_loss
                    best_epoch = epoch
                    best_minibatch = minibatch_index
            if patience <= idx:
                done_looping = True
                break

    print('Saved Model With Respect To Epoch %d , Minibatch %d And Cost Of %f' % \
           (best_epoch, best_minibatch, lowest_cost))

    reconstruct_images = theano.function(
        inputs=[],
        outputs=autoencoder.reconstruction,
        givens={
            imageData: test_images[:100]
        }
    )

    reconstructed_images = reconstruct_images()
    reconstructed_images.reshape(100,28,28)# * 255


    outpath = None
    if parsed.sparsity == 'l1reg':
        outpath = outpath_raw + '/reconstruct_' + str(parsed.l1reg) + '_l1.png'
    else:
        outpath = outpath_raw + '/reconstruct_' + str(parsed.kul_leib_beta) + '_kul_leib.png'

    arraysToImgs(rows=10,colums=10,arr=reconstructed_images,path=outpath,out_shape=(28,28))


def gradient_updates_momentum(cost, params, learning_rate, momentum):
    assert momentum < 1 and momentum >= 0
  
    updates = []
    for param in params:
        #http://cs231n.github.io/neural-networks-3/#sgd
        last_update = theano.shared(param.get_value()*np.cast[theano.config.floatX](0.))   
        delta = - learning_rate * T.grad(cost, param) + \
             momentum * last_update
        updates.append((param, param + delta))
        updates.append((last_update, delta))
        #updates.append((param_update, momentum * param_update + (1. - momentum)*T.grad(cost, param)))
    return updates

if __name__ == '__main__':
    train_autoencoder()