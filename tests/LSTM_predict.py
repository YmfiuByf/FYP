import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm,trange
import datetime
from preprocessing import *
import warnings
import os
warnings.filterwarnings("ignore")

from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
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


def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu(),y_pred_cls.cpu())

n_mfcc, delta = 15, 0
batch_size = 64
hidden_layer_size=1024
linear_size=[512,256,128,64]
dataset = 'IEMOCAP1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

folder = rf"E:\FYP excel\{dataset}_MFCC{n_mfcc}_LSTM_hidden={hidden_layer_size}_linears={linear_size}"
assert os.path.exists(folder)

for model in os.listdir(folder):
    optimal = 0
    if '.pth' in model:
        acc = model.replace('.pth','')[-3:]
    if acc>optimal:
        optimal = acc
        model_path = os.path.join(folder,model)

model = torch.load(model_path)
model = model.to(device)

metric_func = accuracy
metric_name = "accuracy"
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
def loss_func(y_pred,y_true):
    one_hot = torch.zeros((len(y_true),output_size)).to(device)
    for a,b in zip(one_hot,y_true):
        a[b]=1
    return loss_function(y_pred,one_hot)

## data
csv_path = rf"E:\FYP excel\{dataset}_MFCC{n_mfcc}.csv"
df = pd.read_csv(csv_path)
print(df.shape)
if 7 in df['label']:
    df_filtered = df[df['label']!=0]
    df_filtered = df_filtered[df_filtered['label']!=3]
    df_filtered = df_filtered[df_filtered['label']!=6]
    df_filtered = df_filtered[df_filtered['label']!=7]
    df_filtered[df_filtered['label'] == 8] = 3
    df_filtered[df_filtered['label'] == 9] = 6
input_size=n_mfcc*(delta+1)
output_size = 6
num_samples = None#120000
arr = df_filtered.to_numpy()[:,1:]
# del df,df_filtered

# X, _, y, __ = train_test_split(arr[:,3:-1], arr[:,-1], train_size=num_samples,random_state=0)
X=arr[:num_samples,:n_mfcc*(delta+1)]
y=arr[:num_samples,-1]
y=y.astype('int')
length = len(X)
start_t = np.array(df_filtered.iloc[:, 0])
starts = start_t == 0
s = np.arange(len(starts))[starts]

dur = (s[1:]-s[:-1])
dur = np.append(dur,len(starts)-s[-1])
length = np.max(dur)+1
features,labels = [],[]
for i in range(len(s)-1):
    pad = np.zeros([length-dur[i],X.shape[1]])
    feature = np.concatenate([X[s[i]:s[i+1]],pad])
    features.append(feature)
    labels.append(y[s[i]])
pad = np.zeros([length-dur[-1],X.shape[1]])
pad[:, :] = np.nan
feature = np.concatenate([X[s[-1]:len(starts)],pad])
features.append(feature)
labels.append(y[s[-1]])
features = np.array(features,dtype=np.float32)
labels = np.array(labels,dtype=np.int8)

X_train, X_test, y_train, y_test,dur_train,dur_test = train_test_split(features, labels, dur, test_size=0.2,random_state=0)
X_train = torch.from_numpy(X_train).to(device)
y_train = torch.from_numpy(y_train).type(torch.LongTensor).to(device)
X_test = torch.from_numpy(X_test).to(device)
y_test = torch.from_numpy(y_test).type(torch.LongTensor).to(device)
dur_train = torch.from_numpy(dur_train).to(device)
dur_test = torch.from_numpy(dur_test).to(device)

dl_train = DataLoader(TensorDataset(X_train,y_train,dur_train), batch_size=batch_size, shuffle=True)
dl_test = DataLoader(TensorDataset(X_test,y_test,dur_test), batch_size=batch_size, shuffle=True)

def predict(model, dl_test):
    model.eavl()
    y_preds = np.array([])
    for val_step, (features, labels,dur) in enumerate(dl_test, 1):
        with torch.no_grad():
            features = pack_padded_sequence(torch.tensor(features), dur.cpu().type(torch.int64), batch_first=True, enforce_sorted=False).to(device)
            predictions = model(features).cpu().numpy()
            y_pred = np.argmax(predictions,axis=1)
            np.concatenate([y_preds,y_pred])
    return y_preds

def displayCM(y_pred,label_cat):
    assert dataset in ['IEMOCAP9', 'IEMOCAP6', 'SAVEE','IEMOCAP1']
    if dataset == 'IEMOCAP9':
        display_labels = ['neutral', 'sad', 'happy','frustrated', 'angry', 'excited']
    if dataset in ['IEMOCAP6','IEMOCAP1','IEMOCAP']:
        display_labels = ['neural', 'sad', 'frustration', 'anger', 'happy', 'excitment']
    if dataset == 'SAVEE':
        display_labels = ['neural', 'fear', 'anger', 'disgust', 'happiness', 'sad', 'surprise']
    accuracy = metrics.accuracy_score(label_cat, y_pred)
    balanced_accuracy = balanced_accuracy_score(label_cat,y_pred)
    geo_mean = geometric_mean_score(label_cat,y_pred)
    cm = confusion_matrix(label_cat, y_pred, labels=display_labels)
    print(accuracy)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()
    plt.title(f'{dataset}_LSTM_MFCC{n_mfcc},accuracy={accuracy},bal_acc={balanced_accuracy},geo_mean={geo_mean}')
    plt.savefig(fr"E:\FYP eval\{dataset}_LSTM_MFCC{n_mfcc}.png")
    return accuracy

y_pred = predict(model,dl_test=dl_test)
displayCM(y_pred=y_pred,label_cat=y_test.cpu().numpy)

