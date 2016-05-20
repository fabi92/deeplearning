#!/usr/bin/env python

import argparse
import os
import theano
import theano.tensor as T
import numpy as np

from MinibatchKmeans import MinibatchKmeans
from LoadData import loadMNISTUnSplit, loadCIFAR10UnSplit
from Plotter import arraysToImgs, pcaAndPlot

def mini_kmeans(data, n_clusters=20, batchsize=70000, learning_rate=0.001, epochs=1000, threshold=1e-3,  pca=False, init_cluster_vals=None):   
    
    index = T.lscalar()
    t_data = T.matrix('t_data')
    t_dataToCentroids = T.matrix('t_dataToCentroids')
    dataToCentroids = T.matrix('dataToCentroids')

    n_batches = data.get_value(borrow=True).shape[0] // batchsize

    kmeans = MinibatchKmeans(t_data, t_dataToCentroids, n_clusters, 
        (batchsize, data.get_value(borrow=True).shape[1]), learning_rate, init_cluster_vals)


    batchd2centroids = theano.function(
        inputs=[index],
        outputs= kmeans.dataToCentroids,
        givens={
            t_data: data[index * batchsize: (index + 1) * batchsize]
        }
    )

    fulld2centroids = theano.function(
        inputs=[],
        outputs= kmeans.dataToCentroids,
        givens={
            t_data: data
        }
    )

    get_centroids = theano.function(
        inputs=[],
        outputs= kmeans.centroids,
        givens={
        }
    )

    iterate = theano.function(
        inputs=[dataToCentroids, index],
        outputs= kmeans.error,
        updates=kmeans.get_updates(),
        givens={
            t_dataToCentroids: dataToCentroids,
            t_data: data[index * batchsize: (index + 1) * batchsize]
        }
    )

    epoch = 0
    done_looping = False
    old_centroids = None
    while (epoch < epochs) and (not done_looping):
        epoch = epoch + 1
        for idx in range(n_batches):
            centroids = batchd2centroids(idx)
            dataToCentroids = np.zeros([batchsize,n_clusters])
            dataToCentroids[:,centroids] = 1
            dists = iterate(dataToCentroids, idx)
            new_centroids = get_centroids()
        if old_centroids is not None:
            dif = ((old_centroids - new_centroids) ** 2 ).sum(axis=1) ##euclidean dist
            if (dif < threshold).all():
                print('Kmeans has converged')
                print('Centroids Difference = ' + str(dif) + ' < threshold ' + str(threshold))
                done_looping = True
        old_centroids = new_centroids
        print('Average Disance For Epoch %d Amounts To %f' %(epoch, dists))
    
    rows = n_clusters / 10
    shape_eq = int(np.sqrt(data.get_value(borrow=True).shape[1]))
    
    fullDataToCentroids = fulld2centroids()
    centroids = get_centroids()

    if pca:
        pcaAndPlot(data.get_value(borrow=True), fullDataToCentroids, centroids, no_dims = 2)

    arraysToImgs(rows=rows,colums=10,arr=centroids,
        path='results/centroids.png',out_shape=(shape_eq, shape_eq))


if __name__ == '__main__':
    theano.config.exception_verbosity='high'
    ## parses the provided parameters according to the command line input
    parser = argparse.ArgumentParser(prog='Mini Batch Kmeans', conflict_handler='resolve')
    parser.add_argument('-b', '--batchsize', default=20, type=int, required=False, help='Size Of A Single Batch')    
    parser.add_argument('-l', '--learning_rate', default=0.01, type=float, required=False, help='Learning Rate For The Update Operation')    
    parser.add_argument('-e', '--epochs', default=1000, type=int, required=False, help='Size Of A Single Batch') 
    parser.add_argument('-e', '--threshold', default=1e-3, type=float, required=False, help='Threshold For Min Cluster Update') 
    parser.add_argument('-p', '--pcaAndPlot', type=bool, required=False, default=False, help='PCA Reduce to 2')   
    requiredNamed = parser.add_argument_group('required Arguments')
    requiredNamed.add_argument('-p', '--path', type=str, required=True, help='Path To The Training Set')
    requiredNamed.add_argument('-d', '--dataset', type=str, choices=['mnist', 'cifar'])
    requiredNamed.add_argument('-c', '--clusters', type=int, required=True, help='Number Of Cluster Centers')

    parsed = parser.parse_args()

    if parsed.dataset == 'mnist':
        print('Loading Mnist Data')
        (data, labels) = loadMNISTUnSplit(parsed.path, shared=False)
    elif parsed.dataset == 'cifar':
        print('Loading Cifar Data')
        (data, labels) = loadCIFAR10UnSplit(parsed.path, shared=False)
        data = data / 255.

    if not os.path.exists('out'):
        os.makedirs('out')

    # in order to randomly select points as initial cluster centers
    #data = data[(labels == 1) + (labels == 9)]
    valueVec = np.random.randint(data.shape[0], size=parsed.clusters)
    init_cluster_vals = data[valueVec]

    data = theano.shared(np.asarray(data,
                                        dtype=theano.config.floatX),
                                        name='imageData',
                                        borrow=True)
    
    mini_kmeans(data=data, n_clusters=parsed.clusters, batchsize=parsed.batchsize, \
                    learning_rate=parsed.learning_rate, epochs=parsed.epochs, \
                    init_cluster_vals=init_cluster_vals, pca=parsed.pcaAndPlot, threshold=parsed.threshold)