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
from preprocessing import load_model,get_raw,get_raw_SAVEE,load_best_model
import librosa

import pandas as pd
torch.cuda.empty_cache()

dropout_rate = 0.8
kernel_length=[400,800,1600,3200]

test_mode = False
dataset = 'IEMOCAP1'
dataset = 'SAVEE'
folder = rf"E:\FYP excel\{dataset}_1dCNN_kernel={kernel_length}"
if not os.path.exists(folder):
   os.makedirs(folder)
output_size = 6 if dataset=='IEMOCAP1' else 7


if dataset=='IEMOCAP1':
    X_train,X_test,y_train,y_test = get_raw()
elif dataset=='SAVEE':
    X_train,X_test,y_train,y_test = get_raw_SAVEE()
print(len(X_train),len(X_test),len(y_train),len(y_test))
print(torch.unique(y_train),torch.unique(y_test))
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

class my_1dCNN(nn.Module):
    def __init__(self,kernel_length=kernel_length,hidden_layer=[4096,2048,1024],Cin=1,Cout=256):
        super().__init__()
        self.num_kernel = len(kernel_length)
        self.num_layer = len(hidden_layer)
        self.Cout = Cout
        self.convs = nn.ModuleList()
        for KL in kernel_length:
            self.convs.append( nn.Conv1d(in_channels=Cin,out_channels=Cout,kernel_size=KL,stride=int(KL*0.4) ).to(device))
        self.adPooling = nn.AdaptiveMaxPool1d(9)
        self.linears = nn.ModuleList()
        self.relu = nn.ReLU()
        self.last_linear = nn.Linear(hidden_layer[-1],output_size)
        self.input_size = 9*self.Cout * self.num_kernel
        for i in range(self.num_layer):
            if i == 0:
                input = self.input_size
            else:
                input = hidden_layer[i - 1]
            self.linears.append(nn.Linear(input, hidden_layer[i]).to(device))
        self.dropout = nn.Dropout(p=dropout_rate, inplace=False)
        # print(self.convs,self.linears)

    def forward(self,x):
        out = torch.tensor([]).to(device)
        for conv in self.convs:
            # print(conv)
            f = conv(x)
            f = self.adPooling(f).reshape(len(f),9*self.Cout)
            out = torch.cat((out,f),dim=1)
        for linear in self.linears:
            out = self.dropout(out)
            out = linear(out)
            out = self.relu(out)
        out = self.last_linear(out)
        return out




from sklearn.metrics import accuracy_score

def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu(),y_pred_cls.cpu())

def Train_for_list(model, X_train,X_test,y_train,y_test,loss_function, metric_function, optimizer, epochs=150, batch_size=32,model_name='CNN_dd',df_path=folder+f'\\{dataset}_1dCNN.csv'):
    global prob
    optimal = 0.
    global min_loss
    global max_acc
    print(f'min_loss = {min_loss}')
    # for file in os.listdir(r"E:\FYP deep learning model"):
    #     if '.pth' in file and model_name in file:
    #         optimal = file.replace('.pth','')[-5:]
    if not os.path.exists(df_path):
        if dataset == 'SAVEE':
            dfhistory = pd.DataFrame(
                columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name, 'anger', 'disgust', 'fear',
                         'happiness', 'neural', 'sad', 'surprise'])
        elif dataset == 'IEMOCAP1':
            dfhistory = pd.DataFrame(
                columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name, 'neu', 'sad', 'hap', 'fru',
                         'ang', 'exc'])  # ['anger', 'disgust', 'fear', 'happiness', 'neural', 'sad', 'surprise']
        elif dataset == 'RAVDESS':
            dfhistory = pd.DataFrame(
                columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name, 'neu', 'calm', 'hap', 'sad',
                         'ang', 'fear', 'disgust', 'sur'])
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
        for b in range(0,len(X_train),batch_size):
            #print(seq.size(),labels.size())
            optimizer.zero_grad()
            features = X_train[b:b+batch_size]
            y_true = y_train[b:b+batch_size]
            # print(dur)
            if len(features)>1:
                y_pred = torch.tensor([]).to(device)
                for i in range(len(features)):
                    # print(features[i][None,None,:])
                    feature = features[i][None,None,:]
                    feature = torch.tensor(feature).to(device)
                    # print(dur[i])
                    # print(feature.size())
                    y_pred = torch.cat((y_pred,model(feature)),dim=0)
                    # print(y_pred.size())
            else:
                pass
            # print(y_pred.size(),y_true.size())
            # print(len(y_pred),len(y_true))
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
            step+=1

        model.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1
        # print('here',len(X_test),len(y_test))
        for b in range(0,len(X_test),batch_size):

            with torch.no_grad():
                features = X_test[b:b+batch_size]
                # print(features)
                labels = y_test[b:b+batch_size]
                if len(features) > 1:
                    y_pred = torch.tensor([]).to(device)
                    for i in range(len(features)):
                        feature = torch.tensor(features[i][None,None,:]).to(device)
                        # print(dur[i])
                        # print(feature.size())
                        y_pred = torch.cat((y_pred, model(feature)), dim=0)
                        # print(y_pred.size())
                    predictions=y_pred
                else:
                    features = torch.tensor(features).to(device)
                    # print(f'fshape={features.size()}')
                    # features = torch.transpose(features,1,-1)
                    predictions = model(features)
                val_loss = loss_function(predictions, labels)
                val_metric = metric_func(predictions, labels)
            val_step+=1

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
            dfhistory.iloc[epoch-1-output_size:epoch-1, -output_size:] = CM_val.cpu().numpy()

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f," + metric_name + \
               "  = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f")
              % info)

        if loss_sum / step < min_loss:
            min_loss = loss_sum / step
            model_save_path = folder + f'\\{dataset}_CNN1d_{val_metric_sum / val_step:.3f}_loss={min_loss:.3f}.pth'.replace(
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



CNN,min_loss,max_acc,model_list = load_model(folder)
if CNN is None:
    CNN = my_1dCNN().to(device)
# CNN = torch.load(r"E:\FYP excel\IEMOCAP1_encoder_CNN\IEMOCAP1_CNN_MFCC30_402_loss=1.816.pth").to(device)
if test_mode:
    model = load_best_model(folder)
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


optimizer = torch.optim.Adam(CNN.parameters(), lr=1e-4,weight_decay=0.01)

Train_for_list(CNN,X_train,X_test,y_train,y_test,loss_function=loss_function,metric_function=metric_func,optimizer=optimizer,epochs=10000000000)



