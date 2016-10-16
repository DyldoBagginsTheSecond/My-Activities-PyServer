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
    # print("window {}".format(window))
    sum = np.sum(window, axis=0)
    # print("sum {}".format(sum))
    sqr = np.power(sum, 2)
    # print("sqr {}".format(sqr))
    axissum = np.sum(sqr, axis=0)
    # print("axisSum {}".format(axissum))
    sroot = np.sqrt(axissum)
    # print("sroot {}".format(sroot))
    return sroot

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature matrix X.
    
    Make sure that X is an N x d matrix, where N is the number 
    of data points and d is the number of features.
    
    """
    mean = _compute_mean_features(window)
    variance = _compute_variance_features(window)
    magnitude = _compute_magnitude_features(window)

    X = []
    X = np.append(X, mean)
    X = np.append(X, variance)
    X = np.append(X, magnitude)


    return X