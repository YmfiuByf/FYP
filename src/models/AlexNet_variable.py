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

import pandas as pd

pooling = 'max'
n_mfcc = 26
dataset = 'IEMOCAP1'
folder = rf"E:\FYP excel\{dataset}_encoder_CNN_mfcc{n_mfcc}_{pooling}pooling"
if not os.path.exists(folder):
   os.makedirs(folder)


csv_path = r"C:\Users\DELL\Desktop\ComParE_2016+label.csv"
csv_path = r"C:\Users\DELL\Desktop\IEMOCAP_MFCC14+F0.csv"
csv_path = r"C:\Users\DELL\Desktop\IEMOCAP_MFCC14_F0_delta_delta.csv"
csv_path = fr"E:\FYP excel\IEMOCAP1_MFCC{n_mfcc}.csv"

# csv_path = r"C:\Users\DELL\Desktop\SAVEE+MFCC14+F0.csv"
# csv_path = r"C:\Users\DELL\Desktop\IEMOCAP12345_ComParE_2016.csv"
# csv_path = r"C:\Users\DELL\Desktop\SAVEE_ComParE_2016.csv"
df = pd.read_csv(csv_path)
print(np.unique(df['label']))
if 9 in np.array(df['label']):
    print('remove by labels')
    df_filtered = df[df['label']!=0]
    df_filtered = df_filtered[df_filtered['label']!=3]
    df_filtered = df_filtered[df_filtered['label']!=6]
    df_filtered = df_filtered[df_filtered['label']!=7]
    df_filtered['label'][df_filtered['label'] == 8] = 3
    df_filtered['label'][df_filtered['label'] == 9] = 6

output_size = 6

not_allow_to_load= False
CNN,min_loss,max_acc,model_list = load_model(folder)
if CNN is None or not_allow_to_load:
    dropout = 0.8
    CNN = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    # CNN = torch.load(r"E:\FYP excel\IEMOCAP1_MFCC64_AlexNet\IEMOCAP1_AlexNet_MFCC64_652.pth")
    conv = CNN.features[0]
    CNN.features[0] = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(1, 1))
    CNN.classifier[0] = nn.Dropout(p=dropout, inplace=False)
    CNN.classifier[3] = nn.Dropout(p=dropout, inplace=False)
    # CNN.classifier[4] = nn.Linear(4096, 768).to(device)
    CNN.classifier[6] = nn.Linear(4096,6).to(device)
    CNN = CNN.to(device)
    for param, conv_param in zip(CNN.features[0].parameters(), conv.parameters()):
        param = conv_param


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

# # Fixed Length Feature
# l = 45
# cnn_features = []
# labels = []
# i,j = 0,l
# # X = X[:100]
# # y = y[:100]
# while j<length:
#     if starts[j]:
#         i = j
#         j = i+l
#         continue
#     cnn_features.append(X[i:j,:])
#     labels.append(y[i])
#     i+=1
#     j = i+l

# Variable Length Feature
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
features = np.array(features,dtype=np.float32).reshape((len(features),length,n_mfcc,3),order='F')
labels = np.array(labels,dtype=np.int8)
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

X_train, X_test, y_train, y_test,dur_train,dur_test = train_test_split(X, y, dur, test_size=0.2,random_state=0)
X_train = torch.from_numpy(X_train).to(device)
y_train = torch.from_numpy(y_train).type(torch.LongTensor).to(device)
X_test = torch.from_numpy(X_test).to(device)
y_test = torch.from_numpy(y_test).type(torch.LongTensor).to(device)
dur_train = torch.from_numpy(dur_train).to(device)
dur_test = torch.from_numpy(dur_test).to(device)
dl_train = DataLoader(TensorDataset(X_train,y_train,dur_train), batch_size=16, shuffle=True)
dl_test = DataLoader(TensorDataset(X_test,y_test,dur_test), batch_size=16, shuffle=True)

print('loader built')

class myCNN(nn.Module):
    def __init__(self,input_sample, hidden_layer=[1024,512,256,128], kernel_size = [[5,5,64],[3,3,256]], output_size=output_size):
        super().__init__()
        self.output_size = output_size
        self.activation = nn.ReLU()
        self.dense_layers = []
        self.num_hidden = len(hidden_layer)
        self.convs = []
        self.num_kernel = len(kernel_size)
        self.num_layer = len(hidden_layer)
        for i in range(self.num_kernel):
            self.convs.append(nn.Conv2d(in_channels=1 if i==0 else kernel_size[i-1][-1],out_channels=kernel_size[i][-1], kernel_size=kernel_size[i][:-1]).to(device) )
        self.pooling = torch.nn.MaxPool2d([3,3], stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False).to(device)
        tmp = input_sample
        for conv in self.convs:
            tmp = conv(tmp)
            tmp = self.pooling(tmp)
            # print(tmp.size())
        self.size = tmp.size()
        self.input_size = self.size[0]*self.size[1]*self.size[2]

        for i in range(self.num_layer):
            if i==0:
                input = self.input_size
            else:
                input = hidden_layer[i-1]
            self.dense_layers.append(nn.Linear(input, hidden_layer[i]).to(device))

        self.output_layer = nn.Linear(hidden_layer[-1],output_size)

    def forward(self,x):
        for conv in self.convs:
            x = conv(x)
            x = self.pooling(x)
        x = x.reshape(-1,self.input_size)
        for layer in self.dense_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.output_layer(x)
        x = torch.nn.functional.softmax(x,dim=1)
        return x

class my_newCNN(nn.Module):
    def __init__(self,n_mfcc=15,kernel_length=[5,10,15,20,25],hidden_layer=[4096,2048,1024,128,64,6],Cin=3,Cout=256):
        super().__init__()
        self.num_kernel = len(kernel_length)
        self.num_layer = len(hidden_layer)
        self.Cout = Cout
        self.convs = nn.ModuleList()
        for KL in kernel_length:
            self.convs.append( nn.Conv2d(in_channels=Cin,out_channels=Cout,kernel_size=[KL,n_mfcc],stride=KL//3 ).to(device))
        if pooling=='max':
            self.adPooling = nn.AdaptiveMaxPool2d((9,1))
        elif pooling == 'average':
            self.adPooling = nn.AdaptiveAvgPool2d((9, 1))
        self.linears = nn.ModuleList()
        self.input_size = 9*self.Cout * self.num_kernel
        for i in range(self.num_layer):
            if i == 0:
                input = self.input_size
            else:
                input = hidden_layer[i - 1]
            self.linears.append(nn.Linear(input, hidden_layer[i]).to(device))
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        # print(self.convs,self.linears)

    def forward(self,x):
        out = torch.tensor([]).to(device)
        for conv in self.convs:
            # print(conv)
            f = conv(x)
            f = self.adPooling(f).reshape(len(f),9*self.Cout)
            out = torch.cat((out,f),dim=1)
        for linear in self.linears:
            out = linear(out)
            out = self.dropout(out)
        return out




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
        dfhistory.loc[epoch - 1] = info + (np.nan,)*output_size
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

def Train(model, dl_train,dl_test,loss_function, metric_function, optimizer, epochs=150, batch_size=32,model_name='CNN_dd',df_path=folder+f'\\{dataset}_CNN_MFCC{n_mfcc}.csv'):
    global prob
    optimal = 0.
    global min_loss
    global model_list
    global max_acc
    print(f'min_loss = {min_loss}')
    # for file in os.listdir(r"E:\FYP deep learning model"):
    #     if '.pth' in file and model_name in file:
    #         optimal = file.replace('.pth','')[-5:]
    if not os.path.exists(df_path):
        dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name,'neu','sad','hap','fru','ang','exc'])
    else:
        dfhistory = pd.read_csv(df_path)
        dfhistory = dfhistory.iloc[:,1:]
        if len(dfhistory)<1:
            dfhistory = pd.DataFrame(
                columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name, 'neu', 'sad', 'hap', 'fru',
                         'ang', 'exc'])
    print(dfhistory,len(dfhistory))
    # nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_step_freq = 100
    for epoch in range(len(dfhistory)+1,epochs+1):
        model.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        CM_train = torch.zeros((output_size,output_size)).int().to(device)
        CM_val = torch.zeros((output_size, output_size)).int().to(device)
        for step, (features,y_true,dur) in enumerate(dl_train,1):
            #print(seq.size(),labels.size())
            optimizer.zero_grad()
            # print(dur)
            if len(features)>1:
                y_pred = torch.tensor([]).to(device)
                for i in range(len(features)):
                    feature = features[i:i+1, :, :dur[i], :]
                    # print(dur[i])
                    # print(feature.size())
                    y_pred = torch.cat((y_pred,model(feature)),dim=0)
                    # print(y_pred.size())
            else:
                features = features[:,:,:dur,:]
                # print(f'fshape={features.size()}')
                # features = torch.transpose(features,1,-1)
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

        for val_step, (features,labels,dur) in enumerate(dl_test,1):
            with torch.no_grad():
                if len(features) > 1:
                    y_pred = torch.tensor([]).to(device)
                    for i in range(len(features)):
                        feature = features[i:i + 1, :, :dur[i], :]
                        # print(dur[i])
                        # print(feature.size())
                        y_pred = torch.cat((y_pred, model(feature)), dim=0)
                        # print(y_pred.size())
                    predictions=y_pred
                else:
                    features = features[:, :, :dur, :]
                    # print(f'fshape={features.size()}')
                    # features = torch.transpose(features,1,-1)
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
        # if val_metric_sum / val_step > optimal :
        #     optimal  = val_metric_sum / val_step
        if loss_sum / step < min_loss:
            min_loss = loss_sum / step
            model_save_path = folder + f'\\{dataset}_AlexNet_MFCC{n_mfcc}_{val_metric_sum / val_step:.3f}_loss={min_loss:.3f}.pth'.replace(
                '0.', '', 1)
            torch.save(model, model_save_path)
            model_list.append(model_save_path)
            if int(val_metric_sum * 1000 / val_step) >= max_acc:
                max_acc = int(val_metric_sum * 1000 / val_step)
            if len(model_list) > 4:
                delete_path = model_list[0]
                id = delete_path.index('loss')
                acc = int(delete_path[id - 4:id - 1])
                if acc >= max_acc:
                    delete_path = model_list[1]
                    del model_list[1]
                else:
                    del model_list[0]
                if os.path.exists(delete_path):
                    os.remove(delete_path)
        dfhistory.to_csv(df_path)
        print(df_path)
    return


# def __init__(self,input_sample, hidden_layer=[1024,512,128], kernel_size = [[3,3,16],[3,3,16],[3,3,16]], output_size=9):


# def train(model, dl_train,dl_test,loss_function, metric, optimizer, epochs=150, batch_size=32,model_path=r'D:\pycharmProject\FYP\deep learning\CNN',df_path=r"D:\pycharmProject\FYP\deep learning\CNN"):
# CNN = myCNN(X_train[0]).to(device)
# print(CNN(X_train[0:16]).size())
# CNN = torch.load(r"D:\pycharmProject\FYP\deep learning\CNN_dd.pth")

CNN,min_loss,model_list = load_model(folder)
if CNN is None:
    CNN = my_newCNN(n_mfcc=n_mfcc).to(device)
# CNN = torch.load(r"E:\FYP excel\IEMOCAP1_encoder_CNN\IEMOCAP1_CNN_MFCC30_402_loss=1.816.pth").to(device)
print(f'CNN={CNN}')
metric_func = accuracy
metric_name = "accuracy"
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
def loss_func(y_pred,y_true):
    one_hot = torch.zeros((len(y_true),output_size)).to(device)
    for a,b in zip(one_hot,y_true):
        a[b]=1
    return loss_function(y_pred,one_hot)


optimizer = torch.optim.Adam(CNN.parameters(), lr=1e-4)

Train(model=CNN,dl_train=dl_train,dl_test=dl_test,loss_function=loss_function,metric_function=metric_func,optimizer=optimizer,epochs=10000000000)



