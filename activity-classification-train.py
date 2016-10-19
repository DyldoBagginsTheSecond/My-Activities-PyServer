# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:02:58 2016

@author: cs390mb

Assignment 2 : Activity Recognition

This is the starter script used to train an activity recognition
classifier on accelerometer data.

See the assignment details for instructions. Basically you will train
a decision tree classifier and vary its parameters and evalute its
performance by computing the average accuracy, precision and recall
metrics over 10-fold cross-validation. You will then train another
classifier for comparison.

Once you get to part 4 of the assignment, where you will collect your
own data, change the filename to reference the file containing the
data you collected. Then retrain the classifier and choose the best
classifier to save to disk. This will be used in your final system.

Make sure to chek the assignment details, since the instructions here are
not complete.

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import svm
from features import extract_features # make sure features.py is in the same directory
from util import slidingWindow, reorient, reset_vars
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import pickle


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = os.path.join('data', 'activity-data.csv')
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,1], data[i,2], data[i,3]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:1],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)


# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

# you may want to play around with the window and step sizes
window_size = 20
step_size = 20

# sampling rate for the sample data should be about 25 Hz; take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds

feature_names = ["mean X", "mean Y", "mean Z", "variance X", "variance Y", "variance Z", "magnitude", "entropy", "peak X", "peak Y", "peak Z",
                 "max X", "max Y", "max Z", "min X", "min Y", "min Z"]
class_names = ["Stationary", "Walking"]

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

n_features = len(feature_names)

X = np.zeros((0,n_features))
y = np.zeros(0,)

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    # omit timestamp and label from accelerometer window for feature extraction:
    window = window_with_timestamp_and_label[:,1:-1]
    # extract features over window:
    x = extract_features(window)

    # print("feature size {}".format(len(x)))
    # append features:
    X = np.append(X, np.reshape(x, (1,-1)), axis=0)
    # append label:
    y = np.append(y, window_with_timestamp_and_label[10, -1])

print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(y)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Plot data points
#
# -----------------------------------------------------------------------------

# We provided you with an example of plotting two features.
# We plotted the mean X acceleration against the mean Y acceleration.
# It should be clear from the plot that these two features are alone very uninformative.
print("Plotting data points...")
sys.stdout.flush()
plt.figure()
formats = ['bo', 'go']
for i in range(0,len(y),10): # only plot 1/10th of the points, it's a lot of data!
    plt.plot(X[i,6], X[i,7], formats[int(y[i])])

plt.show()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

n = len(y)
n_classes = len(class_names)

# TODO: Train and evaluate your decision tree classifier over 10-fold CV.
# Report average accuracy, precision and recall metrics.

cv = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)
tree1 = DecisionTreeClassifier(criterion="entropy", max_depth=3, max_features=3)
tree2 = DecisionTreeClassifier(criterion="entropy", max_depth=3, max_features=6)
tree3 = DecisionTreeClassifier(criterion="entropy", max_depth=6, max_features=3)
tree4 = DecisionTreeClassifier(criterion="entropy", max_depth=3, max_features=6)

def calcAPR(conf):
    acc = 0.0
    prec = np.array(np.zeros(n_classes))
    rec = np.array(np.zeros(n_classes))
    den_prec = 0
    den_rec = 0
    for j in range(0, n_classes):
        num = (float)(conf[j][j]) # Precision & recall have the same numerator

        for k in range(0, n_classes):
            acc += (float)(conf[k][k])
            den_prec += (float)(conf[k][j]) # Denominator to calculate precision
            den_rec += (float)(conf[j][k]) # Denominator to calculate recall

        if (den_prec != 0):
            if (num == 0): # TP = 0 and FP != 0: precision is 0
                prec[j] = 0.0
            else:
                prec[j] = num/den_prec
        else: # TP = 0 and FP = 0: precision is 1
            prec[j] = 1.0

        if (den_rec != 0):
            if (num == 0): # TP = 0 and FN != 0: recall is 0
                rec[j] = 0.0
            else:
                rec[j] = num/den_rec
        else: # TP = 0 and FN = 0: recall is 1
            rec[j] = 1.0

    acc /= (n_classes*n/10.0)
    # print("Accuracy: {}".format(acc))
    # print("Precision: {}".format(prec))
    # print("Recall: {}".format(rec))
    return [acc, prec, rec]

totalAcc = np.zeros(5)
totalPrec = np.zeros((5, n_classes))
totalRec = np.zeros((5, n_classes))

for i, (train_indexes, test_indexes) in enumerate(cv):
    X_train = X[train_indexes, :]
    y_train = y[train_indexes]
    X_test = X[test_indexes, :]
    y_test = y[test_indexes]

    print("Fold {}".format(i))

    tree1.fit(X_train, y_train)
    y_pred1 = tree1.predict(X_test)
    export_graphviz(tree1, out_file='tree1.dot', feature_names = feature_names)
    conf1 = confusion_matrix(y_test, y_pred1)
    # print conf1
    apr = calcAPR(conf1)
    totalAcc[0] += apr[0]
    totalPrec[0] += np.array(apr[1])
    totalRec[0] += np.array(apr[2])

    tree2.fit(X_train, y_train)
    y_pred2 = tree2.predict(X_test)
    export_graphviz(tree2, out_file='tree2.dot', feature_names = feature_names)
    conf2 = confusion_matrix(y_test, y_pred2)
    # print conf2
    apr = calcAPR(conf2)
    totalAcc[1] += apr[0]
    totalPrec[1] += np.array(apr[1])
    totalRec[1] += np.array(apr[2])

    tree3.fit(X_train, y_train)
    y_pred3 = tree3.predict(X_test)
    export_graphviz(tree3, out_file='tree3.dot', feature_names = feature_names)
    conf3 = confusion_matrix(y_test, y_pred3)
    # print conf3
    apr = calcAPR(conf3)
    totalAcc[2] += apr[0]
    totalPrec[2] += np.array(apr[1])
    totalRec[2] += np.array(apr[2])

    tree4.fit(X_train, y_train)
    y_pred4 = tree4.predict(X_test)
    export_graphviz(tree4, out_file='tree4.dot', feature_names = feature_names)
    conf4 = confusion_matrix(y_test, y_pred4)
    # print conf4
    apr = calcAPR(conf4)
    totalAcc[3] += apr[0]
    totalPrec[3] += np.array(apr[1])
    totalRec[3] += np.array(apr[2])

    # TODO: Evaluate another classifier, i.e. SVM, Logistic Regression, k-NN, etc.
    C = 1.0
    clf = svm.LinearSVC( C=C )
    clf.fit(X_train, y_train)
    y_predclf = clf.predict(X_test)
    confclf = confusion_matrix(y_test, y_predclf)
    # print confclf
    apr = calcAPR(confclf)
    totalAcc[4] += apr[0]
    totalPrec[4] += np.array(apr[1])
    totalRec[4] += np.array(apr[2])

print("Tree 1:")
print("Total average accuracy: {}".format(totalAcc[0]/10))
print("Total average precision: {}".format(totalPrec[0]/10))
print("Total average recall: {}\n".format(totalRec[0]/10))

print("Tree 2:")
print("Total average accuracy: {}".format(totalAcc[1]/10))
print("Total average precision: {}".format(totalPrec[1]/10))
print("Total average recall: {}\n".format(totalRec[1]/10))

print("Tree 3:")
print("Total average accuracy: {}".format(totalAcc[2]/10))
print("Total average precision: {}".format(totalPrec[2]/10))
print("Total average recall: {}\n".format(totalRec[2]/10))

print("Tree 4:")
print("Total average accuracy: {}".format(totalAcc[3]/10))
print("Total average precision: {}".format(totalPrec[3]/10))
print("Total average recall: {}\n".format(totalRec[3]/10))

print("Linear SVM:")
print("Total average accuracy: {}".format(totalAcc[4]/10))
print("Total average precision: {}".format(totalPrec[4]/10))
print("Total average recall: {}\n".format(totalRec[4]/10))


# TODO: Once you have collected data, train your best model on the entire
# dataset. Then save it to disk as follows:

# when ready, set this to the best model you found, trained on all the data:
best_classifier = None
with open('classifier.pickle', 'wb') as f: # 'wb' stands for 'write bytes'
    pickle.dump(best_classifier, f)
