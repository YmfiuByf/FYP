from DBN import *
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

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
signals,length=[],[]
get_signal_numpy(signals,length,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\sentences\\wav')
print(signals[0].shape)

flags,label_cat=[],[]
get_label_4cat(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\dialog\\EmoEvaluation')
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
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

# for elm in X_train:
#     print(elm)
# model = torch.load(r"D:\pycharmProject\FYP\DBN.pth")
# error,v_prob,v = model.reconstruct(X_test[0][None,:])
# print(error,v_prob,v,X_test[0][None,:])
