import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
from tqdm import *
import librosa
import sklearn
from sklearn import metrics
from sklearn.model_selection import GridSearchCV





# load model from hub
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')



from preprocessing import *
print(torch.cuda.is_available())


class svm:
    def __init__(self, toler=0.001, max_iter=100, kernel='linear'):
        self.toler = toler
        self.max_iter = max_iter
        self._kernel = kernel

    # 初始化模型
    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0
        # 将Ei保存在一个列表里
        self.alpha = torch.ones(self.m)
        self.E = torch.tensor([self._e(i) for i in range(self.m)], dtype=torch.float)
        # 错误惩罚参数
        self.C = 1.0

    # kkt条件
    def _kkt(self, i):
        y_g = self._g(self.X[i]) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    # g(x)预测值，输入xi（X[i]）
    def _g(self, xi):
        return (self.alpha * self.Y * self.kernel(self.X, xi)).sum() + self.b

    # 核函数,多项式添加二次项即可
    def kernel(self, X_data, x2, gamma=1, r=0, d=3):
        if len(X_data.shape) > 1:
            res = []
            for x1 in X_data:
                res.append(self.kernel(x1, x2).item())
            return torch.tensor(res, dtype=torch.float)
        else:
            x1 = X_data
            if self._kernel == 'linear':
                return (x1 * x2).sum()
            elif self._kernel == 'poly':
                return (gamma * (x1 * x2).sum() + r) ** d
            return 0

    # E（x）为g(x)对输入x的预测值和y的差
    def _e(self, i):
        return self._g(self.X[i]) - self.Y[i]

    # 初始alpha
    def _init_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)
        for i in index_list:
            if self._kkt(i):
                continue
            E1 = self.E[i]
            # 如果E2是+，选择最小的；如果E2是负的，选择最大的
            if E1 >= 0:
                j = torch.argmin(self.E)
            else:
                j = torch.argmax(self.E)
            return i, j
        # return -1,-1

    # 选择alpha参数
    @staticmethod
    def _compare(_alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    # 训练
    def fit(self, features, labels):
        self.init_args(features, labels)
        for t in range(self.max_iter):
            i1, i2 = self._init_alpha()
            # if i1==-1 and i2==-1:
            #     # 没有找到违背kkt条件的点
            #     return
            # 边界
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])
            E1 = self.E[i1]
            E2 = self.E[i2]
            # eta=K11+K22-2K12
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2], self.X[i2]) - 2 * self.kernel(
                self.X[i1], self.X[i2])
            # 不懂为什么有下面这个判断，如果有读者知道不妨在评论区解释一下
            if eta <= 0:
                continue
            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E2 - E1) / eta
            alpha2_new = self._compare(alpha2_new_unc, L, H)
            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alpha2_new)
            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) - self.Y[
                i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - self.Y[
                i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b
            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2
            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new
            self.E[i1] = self._e(i1)
            self.E[i2] = self._e(i2)

    def predict(self, X_data):
        y_pred = []
        for data in X_data:
            r = (self.Y * self.alpha * self.kernel(self.X, data)).sum()
            y_pred.append(torch.sign(r).item())
        return torch.tensor(y_pred, dtype=torch.float)

    def score(self, X_data, y_data):
        y_pred = self.predict(X_data)
        return (y_pred == y_data).sum() / len(y_data)


#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
signals,length=[],[]
get_signal_numpy(signals,length,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\sentences\\wav')
print(signals[0].shape)

flags,label_cat=[],[]
get_label_4cat_ML(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\dialog\\EmoEvaluation')
# get_label_4cat(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\dialog\\EmoEvaluation')
# get_label_4cat(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\dialog\\EmoEvaluation')
# get_label_4cat(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\dialog\\EmoEvaluation')
# get_label_4cat(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\dialog\\EmoEvaluation')

flags = np.array(flags)
length = np.array(length)
signals = np.array(signals)
idx = flags == 1
signals = signals[idx]
length = length[idx]
label_cat = np.array(label_cat)
y = np.repeat(label_cat,length,axis=0)
X = []
for signal in signals:
    for feature in signal:
        X.append(feature)
X = np.array(X)
print(len(X),len(y))

X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_tv,y_tv, test_size=0.2,random_state=0)
classifiers = []


