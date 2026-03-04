import numpy as np
import pandas as pd
import random
import os.path
import math

from scipy.stats import multivariate_normal as binormal
from scipy.sparse.linalg import eigs
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

#  function for splitting the array into chucks - used ceiling to avoid additional function writing
def n_cross_val(A, n):
    for i in range(0, len(A), n):
        yield (A[:i]+A[i+n:], A[i:i + n])

# Fisher's LDA class
class FLDA:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.obs, self.n = X.shape
        self.classes = np.unique(y)
        self.nclass = len(self.classes)

    # this first splits all the data into their classes and then calls and calculates the Sb and Sw vectors and then finds the eigen vectors
    def build(self):
        self.Xeach = self.split_by_class(self.X, self.y)
        self.compute_all_mean()
        Sw, Sb = self.calculate_scatter()
        evals, evecs = eigs(np.dot(np.linalg.inv(Sw), Sb), k=2, which='LM')
        # print (self.X.shape, evecs.shape)
        return np.dot(self.X, evecs)


    def calculate_scatter(self):
        # within class scatter
        Sw = np.zeros((self.n, self.n))
        diff = self.Xeach[0] - np.array(list(self.mueach[0]) * (self.Xeach[0].shape[0])).reshape(self.Xeach[0].shape)
        for i in range(1, self.nclass):
            c = self.classes[i]
            diff = np.vstack((diff,
                              self.Xeach[c] - np.array(list(self.mueach[c]) * (self.Xeach[c].shape[0])).reshape(
                                  self.Xeach[c].shape)))
        diff = np.matrix(diff)

        for e in range(self.obs):
            Sw += np.dot(diff[e, :].T, diff[e, :])

        # between class
        Sb = np.zeros((self.n, self.n))
        for c in self.classes:
            deltamu = self.mueach[c] - self.mu
            Sb += self.Xeach[c].shape[0] * np.dot(deltamu, deltamu.T)

        # print ("Sw is ", Sw.shape , " and Sb is ", Sb)
        return (Sw, Sb)

    def split_by_class(self, X, y):
        return {c: X[np.where(y == c)[0], :] for c in self.classes}

    #  as written, mueach is the mean of every class and mu is the total mean
    def compute_all_mean(self):
        self.mueach = {c: np.mean(self.Xeach[c], axis=0).reshape(self.n, -1) for c in self.classes}
        self.mu = np.mean(self.X, axis=0).reshape([self.n, -1])


class MultiVariateGaussian:
    def __init__(self, X, y):
        self.X = X + np.random.normal(0, 0.001, (X.shape))
        self.y = y
        self.obs, self.n = X.shape
        assert (self.obs == y.shape[0])
        self.classes = np.unique(y)
        self.nclass = len(self.classes)

    # this predicts the score using the formula as explained in the report
    def predict_score(self, Xt):
        temp = 0
        # print ("predict score starting now")
        for c in self.classes:
            temp += (self.sigma[c] * self.prior[c])

        # for c in self.classes:
        #     print (self.sigma[c])
        # print ("temp is ", temp)
        w1 = []
        w0 = []
        con_inverse = np.linalg.pinv(temp)
        for c in self.classes:
            w1.append(con_inverse.dot(self.mu[c]))
            w0.append(-0.5*(self.mu[c].T).dot((con_inverse).dot(self.mu[c])) + np.log(self.prior[c]))
        # print ("shape of wk is ", w0)
        self.w2 = np.asarray(w0).reshape(self.nclass, 1)
        # print("shape of wk is ", self.w2)
        # print("shape of wk is ", w1)
        self.w3 = np.asarray(w1).reshape(self.nclass, self.X.shape[1])
        # print("SHAPPPEEE of wk is ", self.w3.shape)

        rows, n_t = Xt.shape
        pred = []
        score = np.zeros((rows, self.nclass))
        # print ("rows are" , Xt)
        for ele in Xt:
            # print ("ele is ", ele[:,None])
            pred.append(self.w3.dot(ele[:,None]) + self.w2)
        prediction = np.asarray(pred).reshape(rows, self.nclass)
        # print ((prediction).shape , " bla bla bla")
        return prediction
        # # w = np.zeros((self.nclass, self.X.shape[1]))
        # w1 = []
        # w0 = []
        # con_inverse = np.linalg.pinv(temp)
        # print (con_inverse)
        #
        # for c in self.classes:
        #     w1.append(con_inverse.dot(self.mu[c]))
        #     print ("c ", c , " ", con_inverse.dot(self.mu[c]))
        #     w0.append(-0.5*(self.mu[c].T).dot((con_inverse).dot(self.mu[c])) + np.log(self.prior[c]))
        # w2 = np.asarray(w1).reshape(self.nclass, self.X.shape[1])
        # w3 = np.asarray(w0).reshape(self.nclass, 1)
        # print ("shape 1 " , w3.shape, " bla " , w2.shape)
        # pred = []
        # i = 0
        # acc = 0
        # print (pred)
        # print (self.y)
        # correctness = np.array([yn == y for (yn, y) in zip(pred, self.y)])
        # print ("Correctness is " , (np.sum(correctness) / self.y.size))

    # predicts the class using the scores from the function above
    def predict_class(self, score):
        pred = np.argmax(score, axis=1)
        return np.array([self.classes[i] for i in pred])

    def predict(self, Xt):
        score = self.predict_score(Xt)
        return self.predict_class(score)

    def calc_error(self, ynew, target):
        correctness = np.array([yn == y for (yn, y) in zip(ynew, target)])
        return 1 - (np.sum(correctness) / target.size)

    # I am adding minor random normal values to avoid the singular matrix inverse problem
    def validate(self, Xt, yt):
        Xt += np.random.normal(0, 0.001, (Xt.shape))
        mypredictions = self.predict(Xt)
        return self.calc_error(mypredictions, yt)

    def train(self):
        self.compute_mean_and_sigma()
        self.prior = [np.sum(self.y == c) / self.y.size for c in self.classes]
        # for c in self.classes:
        #     print ("Prior for class ", c , " is ", self.prior[c])

    def build(self):
        self.Xeach = self.split_by_class(self.X, self.y)
        self.train()
        return self

    def split_by_class(self, X, y):
        return {c: X[np.where(y == c)[0], :] for c in self.classes}

    def compute_mean_and_sigma(self):
        self.mu = {c: np.mean(self.Xeach[c], axis=0).reshape(self.n, -1) for c in self.classes}
        self.sigma = {c: (np.cov(self.Xeach[c], bias=True, rowvar=False)) for c in self.classes}
        for c in self.classes:
            self.sigma[c] = np.diag(np.diag(self.sigma[c]))
        # for  c in self.classes:
        #     print ("for class ", c , " mean is ", self.mu[c], " cov is " , self.sigma[c])

#  this is the main function which takes in the user inputs and then outputs the values
def LDA2dGaussGM(filename, num_crossval):
    df = pd.read_csv(filename, header=None)
    data = df.values
    X = data[:, :-1]
    y = data[:, -1]
    # print (X.shape)
    X = X + np.random.normal(0, 0.0001, X.shape)
    if len(np.unique(y)) > 10:
        b = np.median(y)
        f = np.vectorize(lambda x: 0 if x < b else 1)
        y = f(y)
        y = np.reshape(y, [X.shape[0], 1])
    # print (y.shape)
    new_X = FLDA(X, y).build()
    # print (new_X)
    # print(new_X.shape)
    # plt.scatter(new_X[:, 0], new_X[:, 1], c = y, marker='.')
    # plt.show()
    indices = list(range(len(y)))
    random.shuffle(indices)

    # finding out the training errors now
    E = []
    fold = 1

    for (train, test) in n_cross_val(indices, len(indices) // num_crossval):
        Y_train, Y_test, X_train, X_test = y[train], y[test], X[train], X[test]

        # Bi-variate Gaussian classifier
        gaussian = MultiVariateGaussian(X_train, Y_train).build()
        ee = gaussian.validate(X_test, Y_test)
        print("Train error-rate for fold-%s:" % fold, ee)
        fold += 1
        E.append(ee)
    errmu = np.mean(E) * 100
    errsigma = np.std(E)

    print()
    print("Mean train-error %0.2f percent" % errmu)
    print("Std train-error %0.2f percent" % errsigma)

    #finding out the testing errors
    E1 = []
    fold1 = 1

    for (train, test) in n_cross_val(indices, len(indices) // num_crossval):
        Y_train, Y_test, X_train, X_test = y[train], y[test], new_X[train], new_X[test]

        # Bi-variate Gaussian classifier
        gaussian = MultiVariateGaussian(X_train, Y_train).build()
        ee = gaussian.validate(X_test, Y_test)
        print("Test error-rate for fold-%s:" % fold1, ee)
        fold1 += 1
        if fold1 <= 10:
            E1.append(ee)
    errmu = np.mean(E1) * 100
    errsigma = np.std(E1)

    print()
    print("Mean test-error %0.2f percent" % errmu)
    print("Std test-error %0.2f percent" % errsigma)

import sys
# LDA2dGaussGM('digits.csv', 10)
def main(argv=sys.argv):

    if len(argv) == 3:
        LDA2dGaussGM(argv[1], int (argv[2]))
    else:
        print('Usage:  python3 LDA2dGaussGM.py /path/to/dataset.csv num_crossval', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()