import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm,trange
import datetime
from preprocessing import *

print(f'GPU:{torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pandas as pd

class Modified_AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

CNN =  torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
# CNN = torch.load(r"E:\FYP excel\IEMOCAP1_MFCC64_AlexNet\IEMOCAP1_AlexNet_MFCC64_652.pth")
CNN.classifier[6] = nn.Linear(4096,6).to(device)
CNN = CNN.to(device)

# ## planA
# csv_path = r"C:\Users\DELL\Desktop\ComParE_2016+label.csv"
# csv_path = r"C:\Users\DELL\Desktop\IEMOCAP_MFCC14+F0.csv"
# csv_path = r"C:\Users\DELL\Desktop\IEMOCAP_MFCC14_F0_delta_delta.csv"
# csv_path = r"C:\Users\DELL\Desktop\IEMOCAP1_MFCC64_delta_delta.csv"
# csv_path = r"E:\FYP excel\IEMOCAP_MFCC64_dd_resized_reduced.csv"
# # csv_path = r"C:\Users\DELL\Desktop\SAVEE+MFCC14+F0.csv"
# # csv_path = r"C:\Users\DELL\Desktop\IEMOCAP12345_ComParE_2016.csv"
# # csv_path = r"C:\Users\DELL\Desktop\SAVEE_ComParE_2016.csv"
# print('loading dataset')
# df = pd.read_csv(csv_path)
# print('finish loading')
# df_filtered = df[df['label']!=0]
# df_filtered = df_filtered[df_filtered['label']!=3]
# df_filtered = df_filtered[df_filtered['label']!=6]
# df_filtered = df_filtered[df_filtered['label']!=7]
# df_filtered[df_filtered['label'] == 8] = 3
# df_filtered[df_filtered['label'] == 9] = 6
# output_size = 6
# frame_step = 30
# num_samples = None#120000
# arr = df_filtered.to_numpy()[:,1:]
# del df,df_filtered
# # X, _, y, __ = train_test_split(arr[:,3:-1], arr[:,-1], train_size=num_samples,random_state=0)
# X=arr[:num_samples,:-1]
# y=arr[:num_samples,-1]
# y=y.astype('int')
# y = y-1
# print(np.unique(y,return_counts=True))

# ## plan B
# csv_path = r"C:\Users\DELL\Desktop\IEMOCAP1_MFCC64_delta_delta.csv"
# # csv_path = r"C:\Users\DELL\Desktop\SAVEE+MFCC14+F0.csv"
# # csv_path = r"C:\Users\DELL\Desktop\IEMOCAP12345_ComParE_2016.csv"
# # csv_path = r"C:\Users\DELL\Desktop\SAVEE_ComParE_2016.csv"
# df = pd.read_csv(csv_path)
#
# df_filtered = df[df['label']!=0]
# df_filtered = df_filtered[df_filtered['label']!=3]
# df_filtered = df_filtered[df_filtered['label']!=6]
# df_filtered = df_filtered[df_filtered['label']!=7]
# df_filtered[df_filtered['label'] == 8] = 3
# df_filtered[df_filtered['label'] == 9] = 6
# output_size = 6
# frame_step = 10
#
#
# arr = df_filtered.to_numpy()[:,1:]
# del df,df_filtered
# num_samples = None
# # X, _, y, __ = train_test_split(arr[:,3:-1], arr[:,-1], train_size=num_samples,random_state=0)
# X=arr[:num_samples,1:-1]
# y=arr[:num_samples,-1]
# y=y.astype('int')
# length = len(X)
# start_t = arr[:, 0]
# starts = start_t == 0
# # print(starts)
# l = 64
# cnn_features = []
# labels = []
# i,j = 0,l
# # X = X[:100]
# # y = y[:100]
# while j<length:
#     if True in starts[i+1:j]:
#         i += 1 + np.where(starts[i+1:j]==True)[0][-1]
#         j = i+l
#         continue
#     if np.random.rand(1)>0.1:
#         i+= frame_step
#         j = i+l
#         continue
#     cnn_features.append( resize_3d( X[i:j,:].reshape((-1,64,3),order='F'),227,227 ) )
#     labels.append(y[i])
#     i+= frame_step
#     j = i+l
#
# X = np.array(cnn_features,dtype=np.float32)
# print(X.shape)
# y = np.array(labels,dtype=np.int8) - 1
# X = np.moveaxis(X,-1,1)

## plan C
dataset = 'IEMOCAP1'
folder = rf"E:\FYP excel\{dataset}_MFCC64_AlexNet_new2"

if not os.path.exists(folder):
   os.makedirs(folder)
csv_path = r"E:\FYP excel\IEMOCAP_MFCC64_dd_resized_reduced.csv"
# csv_path = r"E:\FYP excel\IEMOCAP_MFCC63F0_dd_resized_reduced.csv"
csv_path = r"E:\FYP excel\IEMOCAP_MFCC64_dd_resized_1200.csv"
print('loading dataset')
num_sample = None
arr = np.array(pd.read_csv(csv_path),dtype=np.float32)[ : , 1:num_sample ].T
print('finish loading')
# arr = arr[ arr[:,-1] != 0 ]
# arr = arr[ arr[:,-1] != 3 ]
# arr = arr[ arr[:,-1] != 6 ]
# arr = arr[ arr[:,-1] != 7 ]
# arr[:,-1][ arr[:,-1] != 8 ] == 3
# arr[:,-1][ arr[:,-1] != 9 ] == 6
output_size = 6

X = arr[:,:-1].reshape((len(arr),227,227,3),order='F')
X = np.moveaxis(X,-1,1)
print(X.shape)
y = arr[:,-1].astype(np.int8)
y -= 1
print(f'unique y={np.unique(y,return_counts=True)}')

# ## test
# X = np.zeros([128,3,227,227],dtype=np.float32)
# y = np.zeros([128],dtype=np.int32)
# # y[:,0]=1
# output_size = 6
# frame_step = 30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
X_train = torch.from_numpy(X_train).to(device)
y_train = torch.from_numpy(y_train).to(device)
X_test = torch.from_numpy(X_test).to(device)
y_test = torch.from_numpy(y_test).to(device)
dl_train = DataLoader(TensorDataset(X_train,y_train), batch_size=64, shuffle=True)
dl_test = DataLoader(TensorDataset(X_test,y_test), batch_size=64, shuffle=True)
print(f'y_train:{torch.unique(y_train,return_counts=True)}')
print(f'y_test:{torch.unique(y_test,return_counts=True)}')

from sklearn.metrics import accuracy_score

def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu(),y_pred_cls.cpu())

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
        # nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print("\n" + "==========" * 8 + "%s" % nowtime)
        if val_metric_sum / val_step > optimal :
            optimal  = val_metric_sum / val_step
            if optimal>0.4:
                torch.save(model,folder+f'\\{dataset}_AlexNet_MFCC64_{optimal:.3f}.pth'.replace('0.','',1))
        dfhistory.to_csv(df_path)
    return

# CNN =  torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)


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
      df_path=folder+f'\\{dataset}_AlexNet_MFCC64.csv')



