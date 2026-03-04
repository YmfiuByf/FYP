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
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import joblib
import Signal_Analysis.features.signal as sig
import sys, os
import warnings
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
import pandas as pd
from preprocessing import *



csv_path=r"C:\Users\DELL\Desktop\ComParE_2016.csv"
features = pd.DataFrame()
features = feature_functionals(features=features, csv_path=csv_path,fp='D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav',smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
))
features = feature_functionals(features=features, csv_path=csv_path,fp='D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\sentences\\wav',smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
))
features = feature_functionals(features=features, csv_path=csv_path,fp='D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\sentences\\wav',smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
))
features = feature_functionals(features=features, csv_path=csv_path,fp='D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\sentences\\wav',smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
))
features = feature_functionals(features=features, csv_path=csv_path,fp='D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\sentences\\wav',smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
))
labels,flags = [],[]
get_label_ML(labels,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\dialog\\EmoEvaluation')
get_label_ML(labels,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\dialog\\EmoEvaluation')
get_label_ML(labels,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\dialog\\EmoEvaluation')
get_label_ML(labels,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\dialog\\EmoEvaluation')
get_label_ML(labels,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\dialog\\EmoEvaluation')
print(len(labels))
num2label =['xxx','neu','sad','fea','fru','ang','sur','dis','hap','exc']
print(labels)
# df = pd.read_csv(csv_path)
df=features
df['label'] = labels
# df.to_csv(csv_path[:-4]+'+label.csv')
df.to_csv(r"C:\Users\DELL\Desktop\IEMOCAP12345_ComParE_2016.csv")
