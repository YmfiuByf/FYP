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

def feature_RAVDESS(fp, csv_path=r"C:\Users\DELL\Desktop\RAVDESS_ComParE_2016.csv", smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
), fs=16000):
    wav_file_path = fp
    orig_wav_files = sorted(os.listdir(wav_file_path))
    num  = 0
    for orig_wav_file in tqdm(orig_wav_files):
        if orig_wav_file[:5] != 'Actor':
            continue
        for wav in sorted(os.listdir(os.path.join(wav_file_path,orig_wav_file))):
            if wav[-3:]!='wav':
                pass
            else:
                signal, sampling_rate = audiofile.read(
                    os.path.join(wav_file_path,orig_wav_file,wav),
                    duration=10,
                    always_2d=True,
                )
                feature = smile.process_signal(
                    signal,
                    sampling_rate
                )
                feature = feature.set_index([pd.Index([wav])])
                feature['label'] = [ wav[7] ]
                if num == 0:
                    features = feature
                else:
                    features = pd.concat([features,feature])
                num+=1
    features.to_csv(csv_path)
    return

def librosa_RAVDESS(n_mfcc=15, fs=16000,test_size=0.2):
    root_path = r"E:\RAVDESS"
    wav_file_path = root_path
    dict = {'n': 1, 'f': 2, 'a': 3, 'd': 4,
            'h': 5}  # num2label3 = ['neural','fear','anger','disgust','happiness','sad','surprise']
    mfccs = pd.DataFrame()
    mfccs_test = pd.DataFrame()
    for orig_wav_file in tqdm(os.listdir(wav_file_path)):
        # print(orig_wav_file)
        if orig_wav_file[:5] != 'Actor':
            continue
        for wav in sorted(os.listdir(os.path.join(wav_file_path, orig_wav_file))):
            if wav[-3:] != 'wav':
                pass
            else:
                wav_path = os.path.join(wav_file_path,orig_wav_file,wav)
                signal, sr = librosa.load(wav_path, sr=fs)
                arr = int(wav[7])

                assert fs == sr
                if n_mfcc>0:
                    feature = np.transpose(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,win_length=400,hop_length=160,n_fft=512))
                    delta = librosa.feature.delta(feature, order=1)
                    delta_delta = librosa.feature.delta(feature, order=2)
                    # print(feature.shape,delta.shape,delta_delta.shape)
                    fd = np.concatenate([feature, delta], axis=1)
                    fdd = np.concatenate([fd, delta_delta], axis=1)
                    # f0 = librosa.pyin(signal,fmin=65,fmax=2093,sr=sr)[0]
                    # fdd = np.concatenate([fdd,f0[:,None]],axis=1)
                    assert fdd.shape[1] == 3 * n_mfcc
                    mfcc = pd.DataFrame(fdd)
                else:
                    # cnt = len(mfccs) + len(mfccs_test)
                    # np.random.seed(cnt)
                    # if np.random.rand() > test_size:
                    #     feature = np.transpose(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,win_length=400,hop_length=160,n_fft=512))
                    #     mfcc = pd.DataFrame(feature)
                    #
                    # else:
                    f0, voiced_flag, voiced_prob = librosa.pyin(signal,fmin=65,fmax=2093,sr=sr,win_length=400,hop_length=160)
                    # delta = librosa.feature.delta(f0, order=1)
                    # delta_delta = librosa.feature.delta(f0, order=2)
                    # fd = np.concatenate([f0,])
                    mfcc = pd.DataFrame(f0[:])
                    # print(mfcc.shape)
                    mfcc['flag'] = voiced_flag[:]
                    mfcc['prob'] = voiced_prob[:]


                mfcc['label'] = np.ones((len(mfcc), 1), dtype=int) * arr
                cnt = len(mfccs) + len(mfccs_test)
                np.random.seed(cnt)
                if np.random.rand() > test_size:
                    mfccs = pd.concat([mfccs, mfcc])
                else:
                    mfccs_test = pd.concat([mfccs_test, mfcc])
    mfccs.to_csv(rf"E:\FYP excel\RAVDESS_MFCC{n_mfcc}_train={1-test_size}.csv")
    mfccs_test.to_csv(rf"E:\FYP excel\RAVDESS_MFCC{n_mfcc}_test={test_size}.csv")
    return mfccs,mfccs_test

librosa_RAVDESS(n_mfcc=0)

# labels,flags = [],[]
# get_label_ML(labels,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\dialog\\EmoEvaluation')
# print(len(labels))
# num2label =['xxx', 'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
# print(labels)
# df = pd.read_csv(csv_path)
# df['label'] = labels
# df.to_csv(csv_path[:-4]+'+label.csv')
