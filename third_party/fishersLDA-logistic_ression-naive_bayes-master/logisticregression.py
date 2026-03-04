import numpy as np
import pandas as pd
import os.path
import math
import sys
from scipy.special import expit as sigmoid
from sklearn.model_selection import train_test_split
from numpy import repeat, dot
from numpy.linalg import inv

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# softmax function
softmax = lambda r : np.exp(r-np.max(r)) / np.sum(np.exp(r-np.max(r)), axis=0)

# class for implementing Logistic regression
class LogisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.all_classes = np.unique(y)

        self.nclass = len(self.all_classes)
        self.number, self.nFeatures = X.shape

        self.W = np.random.normal(0, 0.001, (self.nclass, self.nFeatures))

        # set up weights
        
    # the method used is listed in Wikipedia page
    # the constants used, d and tolerance and hard-coded after running about 20 times to get best (ish) output(error rate)
    def IRLS(self, y, X, cl, iters, deltaa=0.5, tor=0.001):
        # print ("Y vector is ", y.shape, y)
        delW = np.empty(self.W.shape)
        # print (delW.shape, " delW")
        # print (X.shape, " X")
        delta = np.array(repeat(deltaa, self.number)).reshape(1, self.number)
        # print(delta.shape, " delta")
        r = repeat(1, self.number)
        # print ("r is ", r.shape)
        # diagonal elements, all initialized to 1, later updated
        R = np.diag(r)
        # print ("R is ", R.shape)
        Wi = dot(inv(X.T.dot(R).dot(X)), (X.T.dot(R).dot(y)))
        # print ("wi ", Wi.shape)

        # going over the 100 iterations (randomly picked 100)
        for it in range(iters):
            delW = Wi
            delr = abs(y - X.dot(Wi)).T
            r = 1.0 / np.maximum(delta, delr)
            # print("r is ", delta.shape)
            R = np.diag(r[0])
            Wi = dot(inv(X.T.dot(R).dot(X)), (X.T.dot(R).dot(y)))
            t = sum(abs(Wi - delW))
            if t < tor:
                return Wi

    def train(self):
        for i in range(len(self.W)):
            # print ("right now, doing for ", i)
            thisy = np.array([1
                              if c == self.all_classes[i] else 0 for c in self.y])
            self.W[i, :] = self.IRLS(thisy, self.X, self.all_classes[i], 100)

    def predict(self, Xnew):
        ynew = []
        for xnew in Xnew:
            # print (xnew)
            pred = []
            for each in self.all_classes:
                c = int(each)
                pred.append(dot(self.W[c], xnew))
            # print ("predicted is ", pred)
            predicted_with_softmax = softmax(pred)
            # print (predicted_with_softmax.shape)
            match = np.argmax(predicted_with_softmax)
            ynew.append(self.all_classes[match])
        return ynew

    def error123(self, ynew, target):
        correctness = np.array([yn == y for (yn, y) in zip(ynew, target)])
        # print ("Correctness ", np.sum(correctness), target.size)

        return 1 - (np.sum(correctness) / target.size)

    def validate(self, Xnew, Ynew):
        mypredictions = self.predict(Xnew)
        return self.error123(mypredictions, Ynew)

    def build(self):
        # print("Let's start building the logistic regression model.. ")
        self.train()
        return self


# select % of random sub-indices out of the given dataset
def cross_validation(dSet, percent):
    r, _ = dSet.shape
    indices = np.arange(r)
    return np.random.choice(indices, math.ceil(percent / 100 * len(indices)), replace=False)


def logisticRegression(filename, num_splits, train_percent=[10,25,50,75,100]):
    num_splits = int(num_splits)
    df = pd.read_csv(filename, header=None)

    data = df.values
    X = data[:, :-1]
    y = data[:, -1]
    # print (X.shape)
    # print (y.shape)
    X = X + np.random.normal(0, 0.001, X.shape)  # to prevent numerical problem

    if len(np.unique(y)) > 15:
        # if the target values are more than some reasonable no (15), we take that as binary classifier
        b = np.median(y)
        f = np.vectorize(lambda x: 0 if x < b else 1)
        y = f(y)
        y = np.reshape(y, [X.shape[0], 1])

    errormatrix = np.zeros((num_splits, len(train_percent)))
    for i in range(num_splits):
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        # print (X_train.shape)
        Y_train = np.reshape(Y_train, [X_train.shape[0], 1])
        Y_test = np.reshape(Y_test, [X_test.shape[0], 1])

        for j, p in enumerate(train_percent):
            selection = cross_validation(Y_train, p)
            train = X_train[selection, :]
            labels = Y_train[selection]
            model = LogisticRegression(train, labels).build()
            errormatrix[i, j] = model.validate(X_test, Y_test)
        # print()
    errmean = np.mean(errormatrix, axis=0)
    errsigma = np.std(errormatrix, axis=0)

    print("Mean test-error" + str(errmean*100))
    print("Std test-error" + str(errsigma*100))
    # plt.xticks(np.arange(10, 100 + 1, 25))

    # plt.errorbar(train_percent, errmean, errsigma, linestyle='-', marker='^')
    # plt.show()
    return errmean, errsigma

# logisticRegression('digits.csv', 2)
def main(argv=sys.argv):

    if len(argv) == 4:
        train_percent_str = argv[3]
        train_percent = np.array(train_percent_str.split(), dtype=int)
        # print (type(train_percent))
        logisticRegression(argv[1], argv[2], train_percent)
    else:
        print('Usage: python3 logisticregression.py /path/to/dataset.csv num_splits train_percent', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()