import numpy as np
import pandas as pd
import random
import os.path
import math
from scipy.sparse.linalg import eigs

from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#  function for splitting the array into chucks - used ceiling to avoid additional function writing

def cross_validation(A, n):
    for i in range(0, len(A), n):
        yield (A[:i]+A[i+n:], A[i:i + n])

# class for Fisher's LDA
class FLDA_2class:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.obversations, self.n = X.shape


    def build(self):
        self.individual_class = self.split(self.X, self.y)
        # print (self.individual_class[0].shape[0])
        self.mean()
        swee = self.within_class_scatter()
        # print ("shape of sw is ", swee.shape)
        # print ("shape of sheee is ", (self.mean_each[0].T.shape), " bla " , self.mean_each[1].shape)
        self.w = np.dot(np.linalg.inv(swee), (self.mean_each[0].T - self.mean_each[1].T))
        # evals, evecs = eigs(np.dot(np.linalg.inv(swee), (self.mean_each[0].T - self.mean_each[1])), which='LM')
        # print ("evals is ", sorted(ev//ecs))
        # print ("w is ", sorted(self.w))
        return self

    def disc(self):
        return self.w

    def within_class_scatter(self):
        Sw = np.zeros((self.n, self.n))
        df = {}
        for c in [0, 1]:
            # print ("rahul ", list(self.mean_each[c]), "  ", self.individual_class[c])
            inter = np.array(list(self.mean_each[c]) * (self.individual_class[c].shape[0])).reshape(
                self.individual_class[c].shape)
            # print (inter.shape, "   ", inter)
            df[c] = self.individual_class[c] - np.array(list(self.mean_each[c]) * (self.individual_class[c].shape[0])).reshape(
                self.individual_class[c].shape)
        diff = np.concatenate([df[0], df[1]])
        diff = np.matrix(diff)

        for e in range(self.obversations):
            Sw += np.dot(diff[e, :].T, diff[e, :])
        return Sw

    def split(self, X, y):
        return {c: X[np.where(y == c)[0], :] for c in [0, 1]}

    def mean(self):
        self.mean_each = {c: np.mean(self.individual_class[c], axis=0).reshape(-1, self.n) for c in [0, 1]}
        self.mu = np.mean(self.X, axis=0).reshape([-1, self.n])
        # for c in [0,1]:
        #     print ("mean is ", self.mean_each[c])


def LDA1dProjection(filename, num_crossval):
    num_crossval = int (num_crossval)
    df = pd.read_csv(filename, header=None)

    data = df.values
    X = data[:, :-1]
    y = data[:, -1]

    if len(np.unique(y)) > 10:
        b = np.median(y)
        f = np.vectorize(lambda x: 0 if x < b else 1)
        y = f(y)
        y = np.reshape(y, [X.shape[0], 1])


    X_train = X
    Y_train = y
    flda = FLDA_2class(X_train, Y_train).build()
    W = flda.disc()
    # print (W)
    Z = X_train.dot(W)


    Z0 = Z[np.where(Y_train == 0)]
    Z1 = Z[np.where(Y_train == 1)]

    # print("Z is ", Z0)

    plt.subplot(1, 2, 1)
    plt.hist(Z0, bins=20, stacked='False')
    plt.hist(Z1, bins=20, stacked='False')
    plt.title('Data')
    plt.show()
# LDA1dProjection('boston.csv',10)
# nb('digits.csv', 10)
import sys
def main(argv=sys.argv):

    if len(argv) == 3:
        LDA1dProjection(argv[1], argv[2])
    else:
        print('Usage: python3 LDA1dProjection.py /path/to/dataset.csv num_crossval', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()