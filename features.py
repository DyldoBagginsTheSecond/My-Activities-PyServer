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

def _compute_peaks(window):

    import sys
    twoStepsBehindZ = sys.maxint
    twoStepsBehindX = sys.maxint
    twoStepsBehindY = sys.maxint
    oneStepBehindZ = sys.maxint
    oneStepBehindX = sys.maxint
    oneStepBehindY = sys.maxint
    peaksX = 0
    peaksY = 0
    peaksZ = 0
    for x, y, z in window:
        if twoStepsBehindX < oneStepBehindX and oneStepBehindX < x :
            peaksX += 1
        if twoStepsBehindY < oneStepBehindY and oneStepBehindY < y:
            peaksY += 1
        if twoStepsBehindZ < oneStepBehindZ and oneStepBehindZ < z:
            peaksZ += 1
        twoStepsBehindX = oneStepBehindX
        twoStepsBehindY = oneStepBehindY
        twoStepsBehindZ = oneStepBehindZ
        oneStepBehindX = x
        oneStepBehindY = y
        oneStepBehindZ = z

    return [peaksX, peaksY, peaksZ]

def _compute_max(window):
    x = np.max(window[:,0], axis=0)
    y = np.max(window[:,1], axis=0)
    z = np.max(window[:,2], axis=0)
    return [x, y, z]

# def _compute_FFT(window):
#     s = np.sin(window)
#     n_freq = 32
#     sp = np.fft.fft(np.sin(window), n=n_freq)
#     freq = np.fft.fftfreq(n_freq)
#     print freq[sp.argmax()]
#     return freq[sp.argsort()]

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
    peaks = _compute_peaks(window)
    max = _compute_max(window)

    X = []
    X = np.append(X, mean)
    X = np.append(X, variance)
    X = np.append(X, peaks)
    X = np.append(X, max)


    return X