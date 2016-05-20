#!/bin/python

from __future__ import print_function

import argparse
import os
import numpy as np

from LogisticRegressor import loadParams
import matplotlib.pyplot as plt
from Plotter import arraysToImgs

def plot():
    parser = argparse.ArgumentParser(prog='Logistic Regression', conflict_handler='resolve',description = '''\
        Plot Receptive Fields
        ''')
    parser.add_argument('-o', '--output', type=str, default="out", required=False, help='Path To The Output Folder')
    parser.add_argument('-f', '--filename', type=str, default="out", required=False, help='Name Of The Image File')

    requiredNamed = parser.add_argument_group('Required Arguments')
    requiredNamed.add_argument('-m', '--model', type=str, required=True, help='The Previously Trained Model')
    
    parsed = parser.parse_args()

    if not os.path.exists(parsed.output):
        os.makedirs(parsed.output)

    params = loadParams(parsed.model)
    weights = params[0]
    
    weights = weights.T
    dims = weights.shape
    
    rows = dims[0]/10
    shape_eq = int(np.sqrt(dims[1]))

    arraysToImgs(rows=rows,colums=10,arr=weights,path=parsed.output + '/' + parsed.filename, \
        out_shape=(shape_eq, shape_eq))
    
if __name__ == '__main__':
    plot()