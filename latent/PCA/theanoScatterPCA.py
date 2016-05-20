#!/usr/bin/env python

from __future__ import print_function
import sys
import argparse
import os
import numpy as np
import theano
import theano.tensor as T
from itertools import product
from runPCA import runPCA
import matplotlib.pyplot as plt
import LoadData

def theanoScatterPCA(path, dataset):
    if dataset == 'mnist':
        print('Loading Mnist Data')
        (imageData, imageLabels) = LoadData.loadMNISTUnSplit(path, shared=False)
        print(imageData.shape)
    elif dataset == 'cifar':
        print('Loading Cifar Data')
        (imageData, imageLabels) = LoadData.loadCIFAR10UnSplit(path, shared=False)
        imageData = imageData / 255.
    print('Loaded')
    
    print("Computing Scatter Plot")
    labelIds = dict()
    for idx in range(len(imageLabels)):
        if str(imageLabels[idx]) not in labelIds:
            labelIds[str(imageLabels[idx])] = []
        labelIds[str(imageLabels[idx])].append(idx)

    fig, plots = plt.subplots(10, 10)
    fig.set_size_inches(50, 50)
    plt.prism()
    for i, j in product(xrange(10), repeat=2):
        if i > j:
            continue

        idx = labelIds[str(i)] + labelIds[str(j)]
        print('\tCalculating PCA For Classes %d And %d' %(i,j))
        X_transformed = runPCA(data=imageData, elems=idx, components=2)
        Y_ = imageLabels[labelIds[str(i)] + labelIds[str(j)]]
        plots[i, j].scatter(X_transformed[:, 0], X_transformed[:, 1], c=Y_)
        plots[i, j].set_xticks(())
        plots[i, j].set_yticks(())
      
        plots[j, i].scatter(X_transformed[:, 0], X_transformed[:, 1], c=Y_)
        plots[j, i].set_xticks(())
        plots[j, i].set_yticks(())
        if i == 0:
            plots[i, j].set_title(j)
            plots[j, i].set_ylabel(j)
    plt.tight_layout()
    plt.savefig('scatter/' + dataset + ".png")
    print("Computing Scatter Plot Finished")


if __name__ == '__main__':

    theano.config.exception_verbosity='high'
    parser = argparse.ArgumentParser(prog='Principal Component Analysis', conflict_handler='resolve',description = '''\
        This script should enable the user to PCA on the given data set''')

    requiredNamed = parser.add_argument_group('Required Arguments')
    requiredNamed.add_argument('-p', '--path', type=str, required=True, help='Path To The Training Set')
    requiredNamed.add_argument('-d', '--dataset', type=str, required=True,choices=['mnist', 'cifar'])
   
    parsed = parser.parse_args()

    if not os.path.exists('scatter'):
        os.makedirs('scatter')

    theanoScatterPCA(parsed.path, parsed.dataset)