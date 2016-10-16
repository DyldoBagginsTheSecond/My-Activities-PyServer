# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:08:49 2016

@author: cs390mb

This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(window, axis=0)

def _compute_variance_features(window):
    return np.var(window, axis=0)

def _compute_magnitude_features(window):
    sum = np.sum(window, axis=0)
    sqr = np.power(sum, 2)
    axissum = np.sum(sqr, axis=0)
    sroot = np.sqrt(axissum)
    return sroot

def _compute_entropy_features(window):
    hist = np.histogram(window)[0]
    entropy = 0
    for (i, val) in enumerate(hist):
        if hist[i] > 0:
            entropy += hist[i] * np.log2([hist[i]])[0]

    return entropy

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature matrix X.
    
    Make sure that X is an N x d matrix, where N is the number 
    of data points and d is the number of features.
    
    """

    X = []
    X = np.append(X, _compute_mean_features(window))
    X = np.append(X, _compute_variance_features(window))
    X = np.append(X, _compute_magnitude_features(window))
    X = np.append(X, _compute_entropy_features(window))

    return X