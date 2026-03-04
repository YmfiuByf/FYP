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
from librosa_mfcc_feature_extraction import feature_IEMOCAP
from librosa_mfcc_SAVEE import librosa_SAVEE

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

# load model from hub
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')



from preprocessing import *
print(torch.cuda.is_available())


def signal_labelling(features, clf):
    ret = []
    for feature in tqdm(features):
        frame_label = clf.predict(feature)
        label, cnt = np.unique(frame_label, return_counts=True)
        idx = np.argmax(cnt)
        label = label[idx]
        ret.append(label)
    return np.array(ret)


def signal2label(signals, clf):
    ret = []
    for signal in tqdm(signals):
        feature = mfcc_f0(signal)[0][0]
        frame_label = clf.predict(feature)
        label, cnt = np.unique(frame_label, return_counts=True)
        idx = np.argmax(cnt)
        label = label[idx]
        ret.append(label)
    return np.array(ret)

def four_emo(signals,flags,length,label_cat):
    flags = np.array(flags)
    length = np.array(length)
    signals = np.array(signals)
    label_cat = np.array(label_cat)
    idx = flags == 1
    signals = signals[idx]
    length = length[idx]
    label_cat = label_cat[idx]
    return signals,length,label_cat

def train_test_clf(clf,name,original_signal, label_cat, X_train=None, y_train=None,need_training=False,m=13, delta=0, fundamental_frequency=False,output_size=6,starts_test=None):
    if need_training:
        clf.fit(X_train,y_train)
    y_pred,label_cat = signal_labelling2(clf,original_signal,label_cat,starts_test,output_size)
    accuracy = metrics.accuracy_score(label_cat, y_pred)
    balanced_accuracy = balanced_accuracy_score(label_cat,y_pred)
    geo_mean = geometric_mean_score(label_cat,y_pred)
    print((f'clf={clf},accuracy={accuracy},bal_acc={balanced_accuracy},geo_mean={geo_mean}'))
    # print(accuracy)
    joblib.dump(clf,fr"D:\pycharmProject\FYP\SAVEE_{name}_m={m}_delta={delta}_f0={fundamental_frequency}.plk")
    cm = confusion_matrix(label_cat,y_pred,labels=clf.classes_)
    cm = cm/np.sum(cm,axis=0)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['anger', 'disgust', 'fear', 'happiness', 'neural', 'sad', 'surprise'])#['neu','sad','hap','fru','ang','exc'])
    disp.plot()
    plt.title(f'clf={clf},accuracy={accuracy},bal_acc={balanced_accuracy},geo_mean={geo_mean}')
    plt.savefig(fr"D:\pycharmProject\FYP\Figure\SAVEE_{name}_m={m}_delta={delta}_f0={fundamental_frequency}.png")

    diag = np.diagonal(cm)
    precision = diag / np.sum(cm, axis=0)
    recall = diag / np.sum(cm, axis=1)
    precision, recall

    return clf, name, cm, precision, recall, accuracy, balanced_accuracy

def displayCM(clf,y_pred,label_cat):
    accuracy = metrics.accuracy_score(label_cat, y_pred)
    cm = confusion_matrix(label_cat, y_pred, labels=clf.classes_)
    print(accuracy)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['neu','sad','hap','fru','ang','exc'])
    disp.plot()
    return accuracy

def get_best_parameters(parameters,clf_type,X_train,y_train):
    clf = GridSearchCV(clf_type, parameters)
    clf.fit(X_train,y_train)
#     print(clf.cv_results_['params'])
#     print(clf.cv_results_['mean_test_score'])
    return clf.best_estimator_, clf.best_params_


def main(result=[], n=0, m=13, delta=0, fundamental_frequency=False,fp='D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav',p_train=0.01,name=['SVM','KNN','SGD','GNB','DecisionTree','HistGradientBoosting']):
    warnings.filterwarnings("ignore")
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    flags2,label_cat=[],[]
    get_label_4cat_ML(label_cat,flags2,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\dialog\\EmoEvaluation')

    signals,length,flags1=[],[],[]
    get_signal_final(signals,flags1,length,m=m,delta=delta,fundamental_frequency=fundamental_frequency,fp=fp, flags_label = flags2)

    # flags2_test,label_cat_test = [], []
    # get_label_4cat(label_cat_test,flags2_test,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\dialog\\EmoEvaluation')
    # signals_test,flags1_test,length_test = [],[],[]
    # get_mfcc_f0(signals_test,flags1_test,length_test,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\sentences\\wav')


    flags1 = np.array(flags1)
    flags2 = np.array(flags2)
    flags = flags1.astype(bool) & flags2.astype(bool)
    signals,length,label_cat = four_emo(signals,flags,length,label_cat)

    # flags1_test = np.array(flags1_test)
    # flags2_test = np.array(flags2_test)
    # flags_test = flags1_test.astype(bool) & flags2_test.astype(bool)
    # signals_test,length_test, label_cat_test = four_emo(signals_test,flags_test,length_test, label_cat_test)

    y = np.repeat(label_cat,length,axis=0)
    X = []
    original_signal = signals
    for signal in signals:
        for feature in signal:
            X.append(feature)
    X = np.array(X)
    print(len(X),len(y))

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=1-p_train,random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_tv,y_tv, test_size=0.2,random_state=0)
    enablePrint()
    # classifiers = []
    # print(len(X_train))
    # clf_svm1 = sklearn.svm.SVC(decision_function_shape='ovo')
    # clf_svm2 = sklearn.svm.SVC(decision_function_shape='ovr')
    # clf_svm3 = sklearn.svm.LinearSVC()
    # classifiers.append(clf_svm1)
    # classifiers.append(clf_svm2)
    # classifiers.append(clf_svm3)
    #
    # for clf in classifiers:
    #     clf.fit(X_train,y_train)
    #     y_pred = clf.predict(X_val)
    #     accuracy = metrics.accuracy_score(y_val, y_pred)
    #     print(clf,accuracy)
    #
    # parameters = {'kernel':('linear', 'rbf','poly'), 'C':[1, 10],'gamma':[1e-1,1e-2,1e-3,1e-4]}
    # clf = sklearn.svm.SVC()
    # svc = sklearn.svm.SVC()
    # clf = GridSearchCV(svc, parameters)
    # clf.fit(X_train,y_train)
    # print(clf.cv_results_)
    # kernels

    # ['SVM', 'KNN', 'SGD', 'GNB', 'DecisionTree', 'HistGradientBoosting']
    names, cms, precisions, recalls, accuracys, balanced_accuracys,accs = [],[],[],[],[],[],[]
    if 'SVM' in name:
        C, gamma, kernel = 1, 1e-1, 'linear'
        clf = sklearn.svm.SVC(C=C, gamma=gamma, kernel=kernel)
        print('begin training SVM')
        clf = make_pipeline(RandomUnderSampler(random_state=0), clf)
        clf, type, cm, precision, recall, accuracy, balanced_accuracy = train_test_clf(clf,'SVM', original_signal, label_cat, X_train, y_train, need_training=True,m=m, delta=delta, fundamental_frequency=fundamental_frequency)
        acc = metrics.accuracy_score(y_val,clf.predict(X_val))
        for (a,b) in zip([names, cms, precisions, recalls, accuracys, balanced_accuracys,accs], [type, cm, precision, recall, accuracy, balanced_accuracy,acc]):
            a.append(b)
        print('finish_SVM')

    if 'KNN' in name:
        clf = sklearn.neighbors.KNeighborsClassifier()
        parameters = {'weights':('uniform', 'distance'), 'n_neighbors':[5,10,25,int(len(X_train)/10)]}
        clf, params = get_best_parameters(parameters,clf,X_train,y_train)
        print(params,clf)
        clf = make_pipeline(RandomUnderSampler(random_state=0), clf)
        clf, type, cm, precision, recall, accuracy, balanced_accuracy = train_test_clf(clf,'KNN', original_signal, label_cat, X_train, y_train, need_training=True,m=m, delta=delta, fundamental_frequency=fundamental_frequency)
        acc = metrics.accuracy_score(y_val, clf.predict(X_val))
        for (a,b) in zip([names, cms, precisions, recalls, accuracys, balanced_accuracys,accs], [type, cm, precision, recall, accuracy, balanced_accuracy,acc]):
            a.append(b)
        print('finish_KNN')

    if 'SGD' in name:
        clf = sklearn.linear_model.SGDClassifier()
        parameters = {'loss':('hinge','modified_huber','log_loss'),'penalty':('l2','l1')}
        clf, params = get_best_parameters(parameters,clf,X_train,y_train)
        print(params,clf)
        clf = make_pipeline(RandomUnderSampler(random_state=0), clf)
        clf, type, cm, precision, recall, accuracy, balanced_accuracy = train_test_clf(clf,'SGD', original_signal, label_cat, X_train, y_train, need_training=True,m=m, delta=delta, fundamental_frequency=fundamental_frequency)
        acc = metrics.accuracy_score(y_val, clf.predict(X_val))
        for (a,b) in zip([names, cms, precisions, recalls, accuracys, balanced_accuracys,accs], [type, cm, precision, recall, accuracy, balanced_accuracy,acc]):
            a.append(b)
        print('finish_SGD')

    if 'GNB' in name:
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        parameters = {'var_smoothing': [1e-1,1e-2,1e-3,1e-4,1e-5]}
        clf, params = get_best_parameters(parameters, clf, X_train, y_train)
        print(params, clf)
        clf = make_pipeline(RandomUnderSampler(random_state=0),clf)
        clf, type, cm, precision, recall, accuracy, balanced_accuracy = train_test_clf(clf, 'GNB', original_signal, label_cat, X_train, y_train, need_training=True,m=m, delta=delta, fundamental_frequency=fundamental_frequency)
        for (a,b) in zip([names, cms, precisions, recalls, accuracys, balanced_accuracys,accs], [type, cm, precision, recall, accuracy, balanced_accuracy,acc]):
            a.append(b)
        print('finish_GNB')

    if 'DecisionTree' in name:
        clf = sklearn.tree.DecisionTreeClassifier()
        parameters = {'criterion':('gini','entropy', 'log_loss'), 'max_depth': [None,1,2,3,5,10],'max_features':(None,'sqrt','log2')}
        clf, params = get_best_parameters(parameters, clf, X_train, y_train)
        print(params, clf)
        clf, type, cm, precision, recall, accuracy, balanced_accuracy = train_test_clf(clf, 'Tree', original_signal, label_cat, X_train, y_train, need_training=True,m=m, delta=delta, fundamental_frequency=fundamental_frequency)
        acc = metrics.accuracy_score(y_val, clf.predict(X_val))
        for (a,b) in zip([names, cms, precisions, recalls, accuracys, balanced_accuracys,accs], [type, cm, precision, recall, accuracy, balanced_accuracy,acc]):
            a.append(b)
        print('finish_DecisionTree')

    if 'HistGradientBoosting' in name:
        clf = make_pipeline(
            RandomUnderSampler(random_state=0),
            HistGradientBoostingClassifier(random_state=0)
        )
        clf, type, cm, precision, recall, accuracy, balanced_accuracy = train_test_clf(clf, 'IMB_HistGradientBoosting', original_signal, label_cat, X_train, y_train, need_training=True,m=m, delta=delta, fundamental_frequency=fundamental_frequency)
        acc = metrics.accuracy_score(y_val, clf.predict(X_val))
        for (a,b) in zip([names, cms, precisions, recalls, accuracys, balanced_accuracys,accs], [type, cm, precision, recall, accuracy, balanced_accuracy,acc]):
            a.append(b)
        print('finish_IMB_HistGradientBoosting')
    for i in range(len(names)):
        for j in range(0,result.shape[1],2):
            if j<=7:
                result[n+i][j] = precisions[i][j//2]
                result[n+i][j+1] = recalls[i][j//2]
            else:
                result[n+i][-2] = accuracys[i]
                result[n+i][-1] = balanced_accuracys[i]

def signal_labelling2(clf,X,y,starts,output_size):
    s = np.arange(len(starts))[starts]
    y_pred = []
    for i in range(len(s) - 1):
        X_seg = X[s[i]:s[i + 1]]
        # print(len(X_seg))
        prediction = clf.predict(X_seg)
        label, num = np.unique(prediction,return_counts=True)
        label = label[np.argmax(num)]
        label = np.array(label)
        y_pred.append(label)
    #     cnt = np.zeros(output_size)
    #     for mfcc in X_seg:
    #         cnt[clf.predict(mfcc)] += 1
    #     y_pred.append(np.argmax(cnt))
    y_pred = np.array(y_pred)
    y_true = y[s[:-1]]
    return y_pred, y_true



def ML_exp(result=[], n=0, n_mfcc=20, delta=0, fundamental_frequency=False,p_train=0.01,name=['SVM','KNN','SGD','GNB','DecisionTree','HistGradientBoosting']):
    warnings.filterwarnings("ignore")
    test_mode = False

    remove_silence = True
    n_mfcc = n_mfcc
    dataset = 'SAVEE'
    # dataset = 'SAVEE'

    csv_path = fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}.csv"
    csv_path = fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}_train=0.8.csv"
    csv_f0 = fr"E:\FYP excel\{dataset}_MFCC0_train=0.8.csv"
    csv_f1 = fr"E:\FYP excel\{dataset}_MFCC0_test=0.2.csv"
    df_f0 = pd.read_csv(csv_f0)
    df_f1 = pd.read_csv(csv_f1)
    if test_mode:
        csv_path = fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}_test=0.2.csv"
        csv_f0 = fr"E:\FYP excel\{dataset}_MFCC0_test=0.2.csv"
    # csv_path = r"C:\Users\DELL\Desktop\SAVEE+MFCC14+F0.csv"
    # csv_path = r"C:\Users\DELL\Desktop\IEMOCAP12345_ComParE_2016.csv"
    # csv_path = r"C:\Users\DELL\Desktop\SAVEE_ComParE_2016.csv"
    if os.path.exists(csv_path):
        df_train = pd.read_csv(csv_path)
        df_test = pd.read_csv(fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}_test=0.2.csv")
    else:
        func_dict = {'IEMOCAP1': feature_IEMOCAP, 'SAVEE': librosa_SAVEE}
        function = func_dict[dataset]
        for i in range(1, 100):
            if os.path.exists(fr"E:\FYP excel\{dataset}_MFCC{i}_test=0.2.csv"):
                os.remove(fr"E:\FYP excel\{dataset}_MFCC{i}_test=0.2.csv")
            if os.path.exists(fr"E:\FYP excel\{dataset}_MFCC{i}_train=0.8.csv"):
                os.remove(fr"E:\FYP excel\{dataset}_MFCC{i}_train=0.8.csv")
        df_train, df_test = function(n_mfcc)
        if test_mode:
            df = df_test
        else:
            df = df_train
        df_train.to_csv(fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}_train=0.8.csv")
        df_test.to_csv(fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}_test=0.2.csv")
        df.insert(0, '0', np.arange(len(df)))

    print(np.unique(df_train['label']))
    def remove_by_label(df):
        if 9 in np.array(df['label']):
            print('remove by labels')
            df_filtered = df[df['label'] != 0]
            df_filtered = df_filtered[df_filtered['label'] != 3]
            df_filtered = df_filtered[df_filtered['label'] != 6]
            df_filtered = df_filtered[df_filtered['label'] != 7]
            df_filtered['label'][df_filtered['label'] == 8] = 3
            df_filtered['label'][df_filtered['label'] == 9] = 6
        else:
            df_filtered = df
        del df
        return df_filtered
    df_f0 = remove_by_label(df_f0)
    df_f1 = remove_by_label(df_f1)
    voiced, index = voiced_index(df_f0)
    voiced1, index1 = voiced_index(df_f1)
    df_train = remove_by_label(df_train)
    df_test = remove_by_label(df_test)
    dict = {'IEMOCAP1': 6, 'SAVEE': 7, 'RAVDESS': 8}
    output_size = dict[dataset]

    num_samples = None  # 120000
    def df2np(df_train,df_f0,n_mfcc=n_mfcc,delta=delta,fundamental_frequency=fundamental_frequency):
        arr = df_train.to_numpy()
        del df_train
        X = arr[:, 1:1+n_mfcc*(delta+1)]
        if fundamental_frequency:
            X = np.concatenate([X,df_f0.iloc[:,1][:,None]],axis=1)
        y = arr[:, -1]
        y = y.astype('int')
        length = len(X)
        start_t = arr[0][0]
        starts = arr[:, 0] == start_t
        return X,y,starts
    X,y,starts = df2np(df_train,df_f0)
    if remove_silence:
        X = X[voiced]
        y = y[voiced]
        assert len(X) == len(y)
        assert len(X) == len(index)
        starts = index==0
    original_signal,label_cat,starts_test = df2np(df_test,df_f1)
    if remove_silence:
        original_signal = original_signal[voiced1]
        label_cat = label_cat[voiced1]
        starts_test = index1==0

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=1-p_train,random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_tv,y_tv, test_size=0.2,random_state=0)
    enablePrint()
    names, cms, precisions, recalls, accuracys, balanced_accuracys,accs = [],[],[],[],[],[],[]
    if 'SVM' in name:
        C, gamma, kernel = 1, 1e-1, 'linear'
        clf = sklearn.svm.SVC(C=C, gamma=gamma, kernel=kernel)
        print('begin training SVM')
        clf = make_pipeline(RandomUnderSampler(random_state=0), clf)
        clf, type, cm, precision, recall, accuracy, balanced_accuracy = train_test_clf(clf,'SVM', original_signal, label_cat, X_train, y_train, need_training=True,m=m, delta=delta, fundamental_frequency=fundamental_frequency,output_size=output_size,starts_test=starts_test)
        acc = metrics.accuracy_score(y_val, clf.predict(X_val)) 
        for (a,b) in zip([names, cms, precisions, recalls, accuracys, balanced_accuracys,accs], [type, cm, precision, recall, accuracy, balanced_accuracy,acc]):
            a.append(b)
        print('finish_SVM')

    if 'KNN' in name:
        clf = sklearn.neighbors.KNeighborsClassifier()
        parameters = {'weights':('uniform', 'distance'), 'n_neighbors':[5,10,25,int(len(X_train)/10)]}
        clf, params = get_best_parameters(parameters,clf,X_train,y_train)
        print(params,clf)
        clf = make_pipeline(RandomUnderSampler(random_state=0), clf)
        clf, type, cm, precision, recall, accuracy, balanced_accuracy = train_test_clf(clf,'KNN', original_signal, label_cat, X_train, y_train, need_training=True,m=m, delta=delta, fundamental_frequency=fundamental_frequency,output_size=output_size,starts_test=starts_test)
        acc = metrics.accuracy_score(y_val, clf.predict(X_val))
        for (a,b) in zip([names, cms, precisions, recalls, accuracys, balanced_accuracys,accs], [type, cm, precision, recall, accuracy, balanced_accuracy,acc]):
            a.append(b)
        print('finish_KNN')

    if 'SGD' in name:
        clf = sklearn.linear_model.SGDClassifier()
        parameters = {'loss':('hinge','modified_huber','log_loss'),'penalty':('l2','l1')}
        clf, params = get_best_parameters(parameters,clf,X_train,y_train)
        print(params,clf)
        clf = make_pipeline(RandomUnderSampler(random_state=0), clf)
        clf, type, cm, precision, recall, accuracy, balanced_accuracy = train_test_clf(clf,'SGD', original_signal, label_cat, X_train, y_train, need_training=True,m=m, delta=delta, fundamental_frequency=fundamental_frequency,output_size=output_size,starts_test=starts_test)
        acc = metrics.accuracy_score(y_val, clf.predict(X_val))
        for (a,b) in zip([names, cms, precisions, recalls, accuracys, balanced_accuracys,accs], [type, cm, precision, recall, accuracy, balanced_accuracy,acc]):
            a.append(b)
        print('finish_SGD')

    if 'GNB' in name:
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        parameters = {'var_smoothing': [1e-1,1e-2,1e-3,1e-4,1e-5]}
        clf, params = get_best_parameters(parameters, clf, X_train, y_train)
        print(params, clf)
        clf = make_pipeline(RandomUnderSampler(random_state=0),clf)
        clf, type, cm, precision, recall, accuracy, balanced_accuracy = train_test_clf(clf, 'GNB', original_signal, label_cat, X_train, y_train, need_training=True,m=m, delta=delta, fundamental_frequency=fundamental_frequency,output_size=output_size,starts_test=starts_test)
        acc = metrics.accuracy_score(y_val, clf.predict(X_val))
        for (a,b) in zip([names, cms, precisions, recalls, accuracys, balanced_accuracys,accs], [type, cm, precision, recall, accuracy, balanced_accuracy,acc]):
            a.append(b)
        print('finish_GNB')

    if 'DecisionTree' in name:
        clf = sklearn.tree.DecisionTreeClassifier()
        parameters = {'criterion':('gini','entropy', 'log_loss'), 'max_depth': [None,1,2,3,5,10],'max_features':(None,'sqrt','log2')}
        clf, params = get_best_parameters(parameters, clf, X_train, y_train)
        print(params, clf)
        clf, type, cm, precision, recall, accuracy, balanced_accuracy = train_test_clf(clf, 'Tree', original_signal, label_cat, X_train, y_train, need_training=True,m=m, delta=delta, fundamental_frequency=fundamental_frequency,output_size=output_size,starts_test=starts_test)
        acc = metrics.accuracy_score(y_val, clf.predict(X_val))
        for (a,b) in zip([names, cms, precisions, recalls, accuracys, balanced_accuracys,accs], [type, cm, precision, recall, accuracy, balanced_accuracy,acc]):
            a.append(b)
        print('finish_DecisionTree')

    if 'HistGradientBoosting' in name:
        clf = make_pipeline(
            RandomUnderSampler(random_state=0),
            HistGradientBoostingClassifier(random_state=0)
        )
        clf, type, cm, precision, recall, accuracy, balanced_accuracy = train_test_clf(clf, 'IMB_HistGradientBoosting', original_signal, label_cat, X_train, y_train, need_training=True,m=m, delta=delta, fundamental_frequency=fundamental_frequency,output_size=output_size,starts_test=starts_test)
        for (a,b) in zip([names, cms, precisions, recalls, accuracys, balanced_accuracys,accs], [type, cm, precision, recall, accuracy, balanced_accuracy,acc]):
            a.append(b)
        print('finish_IMB_HistGradientBoosting')
    for i in range(len(names)):
        for j in range(0,result.shape[1],2):
            met = 11 if dataset=='IEMOCAP1' else 13
            if j<=met:
                result[n+i][j] = precisions[i][j//2]
                result[n+i][j+1] = recalls[i][j//2]
            else:
                result[n+i][-3] = accuracys[i]
                result[n+i][-2] = balanced_accuracys[i]
                result[n+i][-1] = accs[i]

def save_LLD(m=13, delta=0, fundamental_frequency=False,fp='D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav'):
    warnings.filterwarnings("ignore")
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    flags2,label_cat=[],[]
    get_label_ML(label_cat,flags2,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\dialog\\EmoEvaluation')
    # get_label_ML(label_cat, flags2, 'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\dialog\\EmoEvaluation')
    # get_label_ML(label_cat, flags2, 'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\dialog\\EmoEvaluation')
    # get_label_ML(label_cat, flags2, 'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\dialog\\EmoEvaluation')
    # get_label_ML(label_cat, flags2, 'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\dialog\\EmoEvaluation')

    signals,length,flags1=[],[],[]
    get_signal_final(signals,flags1,length,m=m,delta=delta,fundamental_frequency=fundamental_frequency,fp=fp, flags_label = flags2)
    # get_signal_final(signals, flags1, length, m=m, delta=delta, fundamental_frequency=fundamental_frequency, fp='D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\sentences\\wav',
    #                  flags_label=flags2)
    # get_signal_final(signals, flags1, length, m=m, delta=delta, fundamental_frequency=fundamental_frequency, fp='D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\sentences\\wav',
    #                  flags_label=flags2)
    # get_signal_final(signals, flags1, length, m=m, delta=delta, fundamental_frequency=fundamental_frequency, fp='D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\sentences\\wav',
    #                  flags_label=flags2)
    # get_signal_final(signals, flags1, length, m=m, delta=delta, fundamental_frequency=fundamental_frequency, fp='D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\sentences\\wav',
    #                  flags_label=flags2)


    # flags2_test,label_cat_test = [], []
    # get_label_4cat(label_cat_test,flags2_test,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\dialog\\EmoEvaluation')
    # signals_test,flags1_test,length_test = [],[],[]
    # get_mfcc_f0(signals_test,flags1_test,length_test,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\sentences\\wav')


    flags1 = np.array(flags1)
    flags2 = np.array(flags2)
    flags = flags1.astype(bool) & flags2.astype(bool)
    signals,length,label_cat = four_emo(signals,flags,length,label_cat)

    # flags1_test = np.array(flags1_test)
    # flags2_test = np.array(flags2_test)
    # flags_test = flags1_test.astype(bool) & flags2_test.astype(bool)
    # signals_test,length_test, label_cat_test = four_emo(signals_test,flags_test,length_test, label_cat_test)

    y = np.repeat(label_cat,length,axis=0)
    y = y[:,None]
    X = []
    original_signal = signals
    for signal in signals:
        for feature in signal:
            X.append(feature)
    X = np.array(X)
    df = pd.DataFrame(np.concatenate((X,y),axis=1))
    df.to_csv(rf"C:\Users\DELL\Desktop\FYP\LLD - m={m},f0={fundamental_frequency},delta={delta}.csv")

from itertools import chain
if __name__=='__main__':

    # ms = range(10,21,1)
    # deltass = [0,1,2]
    # fundamental_frequencys = [True]
    # for m in ms:
    #     for delta in deltass:
    #         for fundamental_frequency in fundamental_frequencys:
    #             print(f'm={m},f0={fundamental_frequency},delta={delta}')
    #             save_LLD(m=m,fundamental_frequency=fundamental_frequency,delta=delta)



    dataset = 'SAVEE'
    n_emo = 7 if dataset=='SAVEE' else 6
    name = ['SVM','KNN','SGD','GNB','DecisionTree','HistGradientBoosting']
    # ms = chain(range(10,20,2),range(20,40,3),range(40,101,5))
    ms = range(75,101,5)
    # ms = range(60,101,5)
    cnt = 0
    # for _ in ms:
    #     cnt+=1
    deltass = [0,1,2]
    fundamental_frequencys = [True,False]
    n_row = 100* len(deltass) * len(fundamental_frequencys) * len(name)
    result = np.zeros([n_row,n_emo*2+3]).astype(np.float32)
    row_names = []
    num = 0
    print(n_row,ms,deltass,name)
    print(f'result={result.shape}')
    columns_dict = {'IEMOCAP1':['neu_precision', 'neu_recall', 'sad_precision', 'sad_recall', 'hap_precision',
                                    'hap_recall', 'fru_precision', 'fru_recall', 'ang_precision', 'ang_recall',
                                    'exc_precision', 'exc_recall', 'accuracy', 'balanced_accuracy', 'overall_acc'],
                    'SAVEE':['ang_precision', 'ang_recall', 'dis_precision', 'dis_recall', 'fear_precision',
                                    'fear_recall', 'hap_precision', 'hap_recall', 'neu_precision', 'neu_recall',
                                    'sad_precision', 'sad_recall','sur_precision', 'sur_recall', 'accuracy', 'balanced_accuracy', 'overall_acc']}
    for m in ms:
        for delta in deltass:
            for fundamental_frequency in fundamental_frequencys:
                n = num*len(name)
                print(f'm={m},f0={fundamental_frequency},delta={delta}')
                ML_exp(result=result,n=n, n_mfcc=m,fundamental_frequency=fundamental_frequency,delta=delta,name=name)
                print(result[n:n+len(name)])
                for clf_type in name:
                   row_names.append(f'mfcc={m},f0={fundamental_frequency},delta={delta},clf={clf_type}')
                num+=1
        df1 = pd.DataFrame(result[:len(row_names),:],
                           index=row_names,
                           columns=columns_dict[dataset])
        df1.to_csv(rf"C:\Users\DELL\Desktop\FYP\{dataset}_m={m},f0={fundamental_frequency},delta={delta}.csv")
    print(result)
    df1 = pd.DataFrame(result,
                       index=row_names,
                       columns=['neu_precision','neu_recall','sad_precision','sad_recall','hap_precision','hap_recall','fru_precision','fru_recall','ang_precision','ang_recall','exc_precision','exc_recall','accuracy','balanced_accuracy','overall_acc'])
    df1.to_csv(rf"C:\Users\DELL\Desktop\FYP\{dataset}_m={m},f0={fundamental_frequency},delta={delta}.csv")