import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm,trange
import datetime
import os
print(f'GPU:{torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from preprocessing import load_model
from sklearn.metrics import accuracy_score
import pandas as pd

torch.cuda.empty_cache()

# pooling = 'max'
n_mfcc = 66
dataset = 'IEMOCAP1'
dataset = 'SAVEE'
folder = rf"E:\FYP excel\{dataset}_AlexNetNew_mfcc{n_mfcc}"
if not os.path.exists(folder):
   os.makedirs(folder)


csv_path = fr"E:\FYP excel\IEMOCAP1_MFCC{n_mfcc}.csv"
df = pd.read_csv(csv_path)
print(np.unique(df['label']))
if 9 in df['label']:
    df_filtered = df[df['label']!=0]
    df_filtered = df_filtered[df_filtered['label']!=3]
    df_filtered = df_filtered[df_filtered['label']!=6]
    df_filtered = df_filtered[df_filtered['label']!=7]
    df_filtered['label'][df_filtered['label'] == 8] = 3
    df_filtered['label'][df_filtered['label'] == 9] = 6

output_size = 6 if dataset=='IEMOCAP1' else 7



num_samples = None#120000
arr = df_filtered.to_numpy()
del df,df_filtered
print(f'arr={arr.shape}')
X = arr[:,1:-1]
y = arr[:,-1]
print(np.unique(y))
# X, _, y, __ = train_test_split(arr[:,1:-1].reshape((-1,n_mfcc,3),order='F'), arr[:,-1], train_size=num_samples,random_state=0)
print(f'X={X.shape}')
# X=arr[:num_samples,3:-1]
# print(X.shape)
# y=arr[:num_samples,-1]
y=y.astype('int')
length = len(X)
start_t = arr[0][0]
starts = arr[:,0] == start_t
# print(starts)

# Fixed Length Feature
l = 66
cnn_features = []
labels = []
i,j = 0,l
cnt = 0
index = []
# X = X[:100]
# y = y[:100]
while j<length:
    if starts[j]:
        i = j
        j = i+l
        cnt+=1
        continue
    feature = X[i:j,:].reshape((66, 66, 3), order='F')
    cnn_features.append(feature)
    labels.append(y[i])
    index.append(cnt)
    i+=6
    j = i+l
features = np.array(cnn_features,dtype=np.float32)
print(features.shape)
labels = np.array(labels,dtype=np.int8)
index = np.array(index)


# #Variable Length Feature
# s = np.arange(len(starts))[starts]
# cnn_features,labels = [],[]
# for i in range(len(s)):
#     begin = s[i]
#     if i==len(s)-1:
#         end = len(starts)
#         begin = s[i]
#     else:
#         end = s[i+1]
#     cnn_features.append(X[begin:end,:])
#     labels.append(y[i])

# #Variable length feature 2
# s = np.arange(len(starts))[starts]
# dur = (s[1:]-s[:-1])
# dur = np.append(dur,len(starts)-s[-1])
# length = np.max(dur)+1
# features,labels = [],[]
# for i in range(len(s)-1):
#     pad = np.zeros([length-dur[i],X.shape[1]])
#     feature = np.concatenate([X[s[i]:s[i+1]],pad])
#     features.append(feature)
#     labels.append(y[s[i]])
# pad = np.zeros([length-dur[-1],X.shape[1]])
# pad[:, :] = np.nan
# feature = np.concatenate([X[s[-1]:len(starts)],pad])
# features.append(feature)
# labels.append(y[s[-1]])
# features = np.array(features,dtype=np.float32).reshape((len(features),length,n_mfcc,3),order='F')
# labels = np.array(labels,dtype=np.int8)
print('feature obtained')

X = features
y = labels
X =np.moveaxis(X,-1,1)
print(X.shape)

y = y-1
print(np.unique(y,return_counts=True))

# labels = np.zeros([len(y),9],dtype=np.int32)
# for (l,i) in zip(labels,y):
#     l[i-1] = 1
# y = labels

# X_train, X_test, y_train, y_test,dur_train,dur_test = train_test_split(X, y, dur, test_size=0.2,random_state=0)
X_train, y_train = X, y
X_train = torch.from_numpy(X_train).to(device)
y_train = torch.from_numpy(y_train).type(torch.LongTensor).to(device)
# dur_train = torch.from_numpy(dur_train).to(device)
# dur_test = torch.from_numpy(dur_test).to(device)
# dl_train = DataLoader(TensorDataset(X_train,y_train,dur_train), batch_size=16, shuffle=True)
# dl_test = DataLoader(TensorDataset(X_test,y_test,dur_test), batch_size=16, shuffle=True)
del X,y
dl_train = DataLoader(TensorDataset(X_train,y_train), batch_size=64, shuffle=False)
print('loader built')

model = torch.load(r"E:\FYP excel\IEMOCAP1_AlexNetNew_mfcc66\IEMOCAP1_AlexNet_MFCC66_956_loss=0.095.pth")

def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu(),y_pred_cls.cpu())

loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
metric_function = accuracy
metric_name = "accuracy"

CM_train = torch.zeros((output_size,output_size)).int().to(device)
loss_sum = 0.
metric_sum = 0.
for step, (features, y_true) in enumerate(dl_train, 1):
    y_pred = model(features)
    # print(y_pred.size(),y_true.size())
    loss = loss_function(y_pred, y_true)
    metric = metric_function(y_pred, y_true)
    loss_sum += loss.item()
    metric_sum += metric.item()
    prediction = torch.argmax(y_pred, dim=1)
    tmp = torch.unique(output_size * y_true + prediction, return_counts=True)
    tmp0 = torch.div(tmp[0], output_size, rounding_mode='trunc')
    tmp1 = tmp[0] % output_size
    CM_train[tmp0, tmp1] += tmp[1]
info = (loss_sum / step, metric_sum / step)
print(f'CM_train={CM_train}')
print(("\nloss = %.3f," + metric_name + \
               "  = %.3f")
              % info)

