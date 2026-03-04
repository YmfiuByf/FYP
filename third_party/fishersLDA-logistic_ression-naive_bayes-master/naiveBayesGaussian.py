import numpy as np
import pandas as pd
import os.path
import math
import sys
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from numpy import repeat, dot
from numpy.linalg import inv
from numpy import ndarray
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class NaiveBayes:
    # constructor function to load and initialize some variables
    def __init__(self, X, y):
        self.X = X
        # self.X = X + np.random.normal(0, 0.001, (X.shape))
        self.y = y
        self.obs, self.n = X.shape
        # print ("1. ", self.obs , "2. ", self.n)
        self.classes = np.unique(y)
        self.nclass = len(self.classes)

    # computes the prior, mean and sigma, and is called second after build
    def train(self):
        self.prior = [np.sum(self.y == c) / self.y.size for c in self.classes]
        self.compute_mean()
        self.compute_sigma()
        # self.compute_std()
    # main function which calculates the main vector and does scoring.
    def predict_score(self, Xt):
        temp = 0
        # print ("predict score starting now")
        # for c in self.classes:
        #     print (self.mu[c].shape)

        for c in self.classes:
            # print ("c is ", c)
            temp += (self.sigma[c] * self.prior[c])

        # for c in self.classes:
        #     print (self.sigma[c])
        # print ("temp is ", temp)
        w1 = []
        w0 = []
        con_inverse = np.linalg.pinv(temp)
        for c in self.classes:
            w1.append(con_inverse.dot(self.mu[c]))
            w0.append(-0.5 * (self.mu[c].T).dot((con_inverse).dot(self.mu[c])) + np.log(self.prior[c]))
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
            pred.append(self.w3.dot(ele[:, None]) + self.w2)
        prediction = np.asarray(pred).reshape(rows, self.nclass)
        # print ((prediction).shape , " bla bla bla")
        return prediction

    # is called after the scoring function to select the class it represents
    def predict_class(self, score):
        pred = np.argmax(score, axis=1)
        return np.array([self.classes[i] for i in pred])

    #  outer class for scoring and selecting classes
    def predict(self, Xt):
        score = self.predict_score(Xt)
        return self.predict_class(score)

    def validate(self, Xt, yt):
        # Xt += np.random.normal(0, 0.001, (Xt.shape))
        mypredictions = self.predict(Xt)
        return self.calc_error(mypredictions, yt)

    def calc_error(self, ynew, target):
        correctness = np.array([yn == y for (yn, y) in zip(ynew, target)])
        return 1 - (np.sum(correctness) / target.size)

    # first function to be called, divides the data into separate classes and then calls train to compute other local variables
    def build(self):
        self.Xeach = self.split(self.X, self.y)
        self.train()
        return self
    # splits the classes
    def split(self, X, y):
        return {c: X[np.where(y == c)[0], :] for c in self.classes}

    def compute_mean(self):
        self.mu = {c: np.mean(self.Xeach[c], axis=0).reshape(self.n, -1) for c in self.classes}
        # print(self.mu)

    #  computes covariances and then sets the non diagonals to zero because naive bayes and its independent property
    def compute_sigma(self):
        self.sigma = {c: (np.cov(self.Xeach[c], bias=True, rowvar=False)) for c in self.classes}
        for c in self.classes:
            self.sigma[c] = np.diag(np.diag(self.sigma[c]))

    def compute_std(self):
        self.std = {c: np.std(self.Xeach[c], axis = 0,ddof=1).reshape(self.n, -1) for c in self.classes}
        # print (self.std[0].shape)
        # print (self.std[1].shape)
        # for c in self.classes:
        #     print ("bla bla", (self.std[c]))


# this function is for cross-validation
def select(train, percent):
    count,_ = train.shape
    all_selection = np.arange(count)
    num = len(all_selection)
    return np.random.choice(all_selection, math.ceil(num * (percent / 100)), replace = False)

# the main function which takes in user inputs and does cross validations and handles everything else (data preprocessing and stuff)
def nb(filename, splits, percent = [10,25,50,75,100]):
    splits = int(splits)
    print (type(percent))
    df = pd.read_csv(filename, header= None)
    data = df.values
    X = data[:, :-1]
    # X += np.random.normal(0, 0.001, (X.shape))

    Y = data[:, -1].astype(int)

    print (len(np.unique(Y)))
    if len(np.unique(Y)) > 15:
        median = np.median(Y)
        function1 = np.vectorize(lambda a: 0 if a < median else 1)
        Y = function1(Y)
        Y = np.reshape(Y, [X.shape[0],1])

    ematrix = np.zeros((splits, len(percent)))
    # print(np.unique(Y))

    classes = np.unique(Y)
    # print ("classes are ", classes)
    # for c in classes:
    #     print(Y[np.where(Y == c)[0], :])

    for iter in range(splits):
        print ("Starting the splitting process")
        print ("Will be distributing it in 80-20 blocks")
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y)
        Y_train = np.reshape(Y_train, [X_train.shape[0], 1])
        Y_test = np.reshape(Y_test, [X_test.shape[0], 1])
        for iter1, iter2 in enumerate(percent, 0):
            # print ("iter1 is " , iter1, " and iter2 is ", iter2)
            selected_parts = select(Y_train, iter2)
            # print ("total selection was ", Y_train)
            # print ("selected selection was ", selected_parts)
            # print ("the difference was ", np.setdiff1d(X_train,selected_parts))
            input = X_train[selected_parts, :]
            output = Y_train[selected_parts]
            # print("The output shape is ", output.shape)
            model = NaiveBayes(input, output).build()
            error_from_model = model.validate(X_test, Y_test)
            print ("train percent {} with error percent {}".format(iter2, error_from_model))
            ematrix[iter,iter1] = error_from_model
    errmu = np.mean(ematrix, axis=0)
    errsigma = np.std(ematrix, axis=0, ddof = 0)
    # errsigma1 = np.std(ematrix, axis=0, ddof = 1)

    print("Mean test-error " + str(errmu*100))
    print("Std test-error " + str(errsigma*100))
    plt.errorbar(percent, errmu, errsigma, label='naive bayes', fmt='o--', color='r', capthick=2)
    plt.show()
    # print("Std test-error  1" + str(errsigma1*100))
    return errmu,errsigma
#
# nb('digits.csv', 10)
def main(argv=sys.argv):

    if len(argv) == 4:
        train_percent_str = argv[3]
        train_percent = np.array(train_percent_str.split(), dtype=int)
        # print (type(train_percent))
        nb(argv[1], argv[2], train_percent)
    else:
        print('Usage: python3 naiveBayesGaussian.py /path/to/dataset.csv num_splits train_percent', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()