#!/usr/bin/env python

import theano
import cPickle as pickle
import os
import numpy as np

def loadMNIST(dataset, shared=True):
    with open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    test_set_x, test_set_y = __splitData(test_set)
    valid_set_x, valid_set_y = __splitData(valid_set)
    train_set_x, train_set_y = __splitData(train_set)   

    if not shared:
        return (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)

    test_set_x, test_set_y = shared_dataset((test_set_x, test_set_y))
    valid_set_x, valid_set_y = shared_dataset((valid_set_x, valid_set_y))
    train_set_x, train_set_y = shared_dataset((train_set_x, train_set_y))

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)

def loadMNISTUnSplit(dataset, shared=True):
    (train_images, train_labels), (validation_images, validation_labels), \
         (test_images, test_labels) = loadMNIST(dataset, False)

    fullSetImgs = np.concatenate((train_images, validation_images, test_images), axis=0)
    fullSetLabels = np.concatenate((train_labels, validation_labels, test_labels), axis=0)
    if shared:
        return shared_dataset((fullSetImgs, fullSetLabels))
    
    return (fullSetImgs, fullSetLabels)

def loadCIFAR10UnSplit_Color(path, shared=True, shuffle=False):
    labels = np.zeros(60000, dtype='int32')
    images = np.zeros((60000, 32, 32, 3), dtype='float64')
    
    for subdir, dirs, files in os.walk(path):
        n_loaded = 0
        for file in files:
            if file.startswith("data"): 
                filepath = subdir + os.sep + file
                fo = open(filepath, 'rb')
                data = pickle.load(fo)
                fo.close()

                assert data['data'].dtype == np.uint8
                images[n_loaded:n_loaded + 10000] = data['data'].reshape(10000, 3,32,32).transpose(0,2,3,1)# np.reshape(data['data'], (10000,32,32,3), order='F')
                labels[n_loaded:n_loaded + 10000] = data['labels']
                n_loaded += 10000
            elif file.startswith('test'):
                filepath = subdir + os.sep + file
                fo = open(filepath, 'rb')
                data = pickle.load(fo)
                fo.close()

                assert data['data'].dtype == np.uint8
                images[50000:60000] = data['data'].reshape(10000, 3,32,32).transpose(0,3,2,1)#data['data'].reshape(10000, 32, 32, 3)
                labels[50000:60000] = data['labels']
    if shuffle:
        img_lables = zip(images, labels)
        np.random.shuffle(img_lables)
        images, labels = zip(*img_lables)

    if shared:
        images, labels = shared_dataset((images, labels), True)
    return images, labels

def loadCIFAR10UnSplit(path, shared=True, shuffle=False):
    labels = np.zeros(60000, dtype='int32')
    grey = np.zeros((60000, 32 * 32), dtype='float64')
    
    for subdir, dirs, files in os.walk(path):
        n_loaded = 0
        for file in files:
            if file.startswith("data"): 
                filepath = subdir + os.sep + file
                fo = open(filepath, 'rb')
                data = pickle.load(fo)
                fo.close()

                assert data['data'].dtype == np.uint8
                def conv2grey(X):
                    return np.dot(X.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)[...,:3], [0.299, 0.587, 0.114])
                #print(grey(data['data']).shape)
                grey[n_loaded:n_loaded + 10000] = conv2grey(data['data']).reshape(10000, 32*32)
                labels[n_loaded:n_loaded + 10000] = data['labels']
                n_loaded += 10000
            elif file.startswith('test'):
                filepath = subdir + os.sep + file
                fo = open(filepath, 'rb')
                data = pickle.load(fo)
                fo.close()

                assert data['data'].dtype == np.uint8
                def conv2grey(X):
                    return np.dot(X.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)[...,:3], [0.299, 0.587, 0.114])
                #print(grey(data['data']).shape)
                grey[50000:60000] = conv2grey(data['data']).reshape(10000, 32*32)
                labels[50000:60000] = data['labels']
    if shuffle:
        grey_lables = zip(grey, labels)
        np.random.shuffle(grey_lables)
        grey, labels = zip(*grey_lables)

    if shared:
        grey, labels = shared_dataset((grey, labels), True)
    return grey, labels

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = __splitData(data_xy)
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX),
                                        name='imageData',
                                        borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX),
                                        name='imageLabels',
                                        borrow=borrow)
    return shared_x, theano.tensor.cast(shared_y, 'int32')

def __splitData(dataset):
    data, labels = dataset
    return data, labels