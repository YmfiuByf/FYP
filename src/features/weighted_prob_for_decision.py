import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm,trange
import datetime
from preprocessing import *
from sklearn.metrics import accuracy_score

def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu(),y_pred_cls.cpu())


print(f'GPU:{torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pandas as pd

class weighted_prob(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        # self.conv = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=[1,6])
        w = torch.tensor([1.7, 1.1, 0.8, 1.7, 0.95, 1.3]).float()
        self.w = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.softmax(x[:,:,:])
        # x = self.conv(x)
        # print(x.size())
        x = x * self.w
        x = torch.sum(x,axis=-2)
        x = x.squeeze()
        # x = torch.argmax(x,axis=-1)
        # print(x.size())
        return x

dataset = 'IEMOCAP1'
# dataset = 'SAVEE'

folder = rf"E:\FYP excel\{dataset}_FollowingAlexNet"
if not os.path.exists(folder):
   os.makedirs(folder)

csv_path = rf"E:\FYP excel\IEMOCAP2_AlexNet_6.csv"
df = pd.read_csv(csv_path)
print(df.shape)
print(np.unique(np.array(df['label'])))
df_filtered = df[df['label']!=0]
if 9 in np.unique(np.array(df['label'])):
    df_filtered = df_filtered[df_filtered['label']!=3]
    df_filtered = df_filtered[df_filtered['label']!=6]
    df_filtered = df_filtered[df_filtered['label']!=7]
    df_filtered['label'][df_filtered['label'] == 8] = 3
    df_filtered['label'][df_filtered['label'] == 9] = 6

input_size= 7 if dataset=='SAVEE' else 6
output_size = 7 if dataset=='SAVEE' else 6
num_samples = None#120000

arr = df_filtered.to_numpy()[:,2:] # 1 for normal, 2 for 4096
# del df,df_filtered
# X, _, y, __ = train_test_split(arr[:,3:-1], arr[:,-1], train_size=num_samples,random_state=0)
X=arr[:num_samples,:input_size]
y=arr[:num_samples,-1]
y=y.astype('int')
y-=1
print(f'X_shape={X.shape}')
# print(np.unique(y,return_counts=True))
length = len(X)
start_t = np.array(df_filtered.iloc[:, 1])  # 0 for normal, 1 for 4096
starts = start_t == 0
print(f'starts={starts}')
s = np.arange(len(starts))[starts]
# print(f's={s[-1]},len_starts={len(starts)}')
# pos = np.where(y == 3)
# print(start_t[7910],start_t[7911],pos)

dur = (s[1:]-s[:-1])
dur = np.append(dur,len(starts)-s[-1])
length = np.max(dur)+1
features,labels = [],[]
prediction = np.zeros(6)
predictions = []
for i in range(len(s)-1):
    pad = np.zeros([length-dur[i],X.shape[1]])
    feature = np.concatenate([X[s[i]:s[i+1]],pad])
    features.append(feature)
    pred = np.argmax(X[s[i]:s[i+1]],axis=-1)
    pred = np.unique(pred,return_counts=True)
    for j in range(len(pred[0])):
        prediction[pred[0][j]] = pred[1][j]
    prediction = prediction/np.sum(prediction)
    predictions.append(prediction)
    # print(y[s[i]])
    labels.append(y[s[i]])
pad = np.zeros([length-dur[-1],X.shape[1]])
# pad[:, :] = np.nan
feature = np.concatenate([X[s[-1]:len(starts)],pad])
features.append(feature)
labels.append(y[s[-1]])
features = np.array(features,dtype=np.float32)
labels = np.array(labels,dtype=np.int8)
print(features.shape,labels.shape,np.unique(labels,return_counts=True),f'output size={output_size}')
# print(np.unique(labels,return_counts=True))

X_train, X_test, y_train, y_test,dur_train,dur_test = train_test_split(features, labels, dur, test_size=0.2,random_state=0)
print(np.unique(y_train,return_counts=True))
X_train = torch.from_numpy(X_train).to(device)
y_train = torch.from_numpy(y_train).type(torch.LongTensor).to(device)
X_test = torch.from_numpy(X_test).to(device)
y_test = torch.from_numpy(y_test).type(torch.LongTensor).to(device)
dur_train = torch.from_numpy(dur_train).to(device)
dur_test = torch.from_numpy(dur_test).to(device)

batch_size = 64
dl_train = DataLoader(TensorDataset(X_train,y_train), batch_size=batch_size, shuffle=True)
dl_test = DataLoader(TensorDataset(X_test,y_test), batch_size=batch_size, shuffle=True)
def train(model, dl_train,dl_test,loss_function, metric_function, optimizer, epochs=150, batch_size=32,model_name='CNN_dd',df_path=r"D:\pycharmProject\FYP\deep learning\CNN"):
    global prob
    optimal = 0.
    # for file in os.listdir(r"E:\FYP deep learning model"):
    #     if '.pth' in file and model_name in file:
    #         optimal = file.replace('.pth','')[-5:]
    dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name,'neu','sad','hap','fru','ang','exc'])
    # nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_step_freq = 100
    for epoch in range(1,epochs+1):
        model.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        CM_train = torch.zeros((output_size,output_size)).int().to(device)
        CM_val = torch.zeros((output_size, output_size)).int().to(device)
        for step, (features,y_true) in enumerate(dl_train,1):
            #print(seq.size(),labels.size())
            optimizer.zero_grad()
            y_pred = model(features)
            # print(y_pred.size(),y_true.size())
            loss = loss_function(y_pred, y_true)
            loss.backward()
            optimizer.step()
            metric = metric_function(y_pred,y_true)

            prediction = torch.argmax(y_pred,dim=1)
            tmp = torch.unique(output_size * y_true + prediction, return_counts=True)
            tmp0 = torch.div(tmp[0], output_size, rounding_mode='trunc')
            tmp1 = tmp[0] % output_size
            CM_train[tmp0, tmp1] += tmp[1]

            loss_sum += loss.item()
            metric_sum += metric.item()
            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
                      (step, loss_sum / step, metric_sum / step))

        model.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features, labels) in enumerate(dl_test, 1):
            with torch.no_grad():
                predictions = model(features)
                val_loss = loss_function(predictions, labels)
                val_metric = metric_func(predictions, labels)

            val_loss_sum += val_loss.item()
            val_metric_sum += val_metric.item()

            prediction = torch.argmax(predictions,dim=1)
            tmp = torch.unique(output_size * labels + prediction, return_counts=True)
            tmp0 = torch.div(tmp[0], output_size, rounding_mode='trunc')
            tmp1 = tmp[0] % output_size
            CM_val[tmp0, tmp1] += tmp[1]

        print(f'CM_train={CM_train},\nCM_val={CM_val}')
        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum / step, metric_sum / step,
                val_loss_sum / val_step, val_metric_sum / val_step)
        dfhistory.loc[epoch - 1] = info + (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
        if epoch%10 == 0:
            dfhistory.iloc[epoch-7:epoch-1, -6:] = CM_val.cpu().numpy()

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f," + metric_name + \
               "  = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f")
              % info)
        print(f'w={model.w}')
        # nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print("\n" + "==========" * 8 + "%s" % nowtime)
        if val_metric_sum / val_step > optimal :
            optimal  = val_metric_sum / val_step
            if optimal>0.4:
                torch.save(model,folder+f'\\{dataset}_FollowingAlexNet_{optimal:.3f}.pth'.replace('0.','',1))
        dfhistory.to_csv(df_path)
    return

# CNN =  torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

CNN = weighted_prob()
CNN = CNN.to(device)

metric_func = accuracy
metric_name = "accuracy"
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
def loss_func(y_pred,y_true):
    one_hot = torch.zeros((len(y_true),output_size)).to(device)
    for a,b in zip(one_hot,y_true):
        a[b]=1
    return loss_function(y_pred,one_hot)


optimizer = torch.optim.Adam(CNN.parameters(), lr=1e-7)
# optimizer = optimizer.to(device)

train(model=CNN,dl_train=dl_train,dl_test=dl_test,loss_function=loss_func,
      metric_function=metric_func,optimizer=optimizer,epochs=10000,model_name='AlexNet',
      df_path=folder+f'\\{dataset}_FollowingAlexNet.csv')

