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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



import pandas as pd
from sklearn.metrics import accuracy_score

def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu(),y_pred_cls.cpu())
print(f'GPU:{torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# lr,dropout,weight_decay = 1e-3, 0.5, 0.1
lr,dropout,weight_decay = 1e-5, 0.7, 0.2
# lr,dropout,weight_decay = 1e-6, 0.8, 0.5
p=1.5
n_mfcc, delta = 100, 2
batch_size = 16
hidden_layer_size=128
linear_size=[512,512]
num_layer = 2
# linear_size=[16,16,8]
dataset = 'IEMOCAP1'
dataset = 'SAVEE'

input_size=n_mfcc*(delta+1)
# input_size=6
output_size = 7 if dataset=='SAVEE' else 6
num_samples = None#120000
test_mode = False
def train(model, dl_train,dl_test,loss_function, metric_function, optimizer, epochs=10000, batch_size=batch_size,model_name='LSTM',df_path=r"D:\pycharmProject\FYP\deep learning\LSTM1.csv",min_loss=10000.,model_list=[],max_acc=0.):
    global prob
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
        dfhistory = dfhistory.iloc[:, 1:]
        if len(dfhistory) < 1:
            dfhistory = pd.DataFrame(
                columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name, 'neu', 'sad', 'hap', 'fru',
                         'ang', 'exc'])
    print(dfhistory, len(dfhistory))
    log_step_freq = 100
    for epoch in range(len(dfhistory)+1,epochs+1):
        model.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        CM_train = torch.zeros((output_size, output_size)).int().to(device)
        CM_val = torch.zeros((output_size, output_size)).int().to(device)
        for step, (features,y_true,dur) in enumerate(dl_train,1):
            #print(seq.size(),labels.size())
            optimizer.zero_grad()
            features = pack_padded_sequence(features, dur.cpu().type(torch.int64),batch_first=True,enforce_sorted=False).to(device)
            y_pred = model(features)
            # print(y_pred.size(),y_true.size())
            loss = loss_function(y_pred, y_true)
            loss.backward()
            optimizer.step()
            metric = metric_function(y_pred,y_true)

            prediction = torch.argmax(y_pred, dim=1)
            tmp = torch.unique(output_size * y_true + prediction, return_counts=True)
            tmp0 = torch.div(tmp[0], output_size, rounding_mode='trunc')
            tmp1 = tmp[0] % output_size
            CM_train[tmp0, tmp1] += tmp[1]

            loss_sum += loss.item()
            metric_sum += metric.item()
            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
                      (step, loss_sum / step, metric_sum / step))
        print(f'CM_train={CM_train}')
        model.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features, labels,dur) in enumerate(dl_test, 1):
            with torch.no_grad():
                features = pack_padded_sequence(torch.tensor(features), dur.cpu().type(torch.int64), batch_first=True, enforce_sorted=False).to(device)
                predictions = model(features)
                val_loss = loss_function(predictions, labels)
                val_metric = metric_func(predictions, labels)

            val_loss_sum += val_loss.item()
            val_metric_sum += val_metric.item()

            prediction = torch.argmax(predictions, dim=1)
            tmp = torch.unique(output_size * labels + prediction, return_counts=True)
            tmp0 = torch.div(tmp[0], output_size, rounding_mode='trunc')
            tmp1 = tmp[0] % output_size
            CM_val[tmp0, tmp1] += tmp[1]

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum / step, metric_sum / step,
                val_loss_sum / val_step, val_metric_sum / val_step)
        dfhistory.loc[epoch - 1] = info + (np.nan,) * output_size
        if epoch % 10 == 0:
            dfhistory.iloc[epoch - 1 - output_size:epoch - 1, -output_size:] = CM_val.cpu().numpy()

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f," + metric_name + \
               "  = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f")
              % info)
        # nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print("\n" + "==========" * 8 + "%s" % nowtime)
        if loss_sum / step < min_loss or int(val_metric_sum * 1000 / val_step) > max_acc:
            min_loss = loss_sum / step
            model_save_path = folder + f'\\{dataset}_LSTM_MFCC{n_mfcc}_{val_metric_sum / val_step:.3f}_loss={min_loss:.3f}.pth'.replace(
                '0.', '', 1)
            torch.save(model, model_save_path)
            model_list.append(model_save_path)
            if int(val_metric_sum * 1000 / val_step) >= max_acc:
                max_acc = int(val_metric_sum * 1000 / val_step)
                if model_save_path not in model_list:
                    torch.save(model, model_save_path)
                    model_list[0] = model_save_path
            if len(model_list) > 3:
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
    return

class LSTM(nn.Module):
    def __init__(self, input_size=input_size, hidden_layer_size=hidden_layer_size, linear_size=linear_size, output_size=output_size, batch_size=batch_size,num_layer=num_layer):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size,num_layers=num_layer).to(device)
        self.dense_layers = nn.ModuleList()
        for i in range(len(linear_size)):
            in_size = linear_size[i-1]
            out_size = linear_size[i]
            if i ==0:
                in_size = hidden_layer_size
            if i==len(linear_size)-1:
                out_size = output_size
            self.dense_layers.append(nn.Dropout(p=dropout, inplace=False))
            self.dense_layers.append(nn.Linear(in_size,out_size).to(device))
            if i!=len(linear_size)-1:
                self.dense_layers.append(nn.ReLU())
            else:
                self.dense_layers.append(nn.Softmax())
            # else:
            #     self.dense_layers.append(nn)


        self.hidden_cell = (torch.zeros(1,batch_size,self.hidden_layer_size),
                            torch.zeros(1,batch_size,self.hidden_layer_size))

    def forward(self, input_seq):
        # ##lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        # lstm_out, self.hidden_cell = self.lstm(input_seq,self.hidden_cell)
        lstm_out, self.hidden_cell = self.lstm(input_seq)
        # ##predictions = self.linear(lstm_out.view(len(input_seq), -1))
        # pred = self.hidden_cell[0][0].squeeze()
        pred,lens = pad_packed_sequence(lstm_out)
        lens = lens.to(device)
        pred = torch.linalg.norm(pred,dim=0,ord=p)
        pred = torch.mul(pred.T,(lens)**(-1/p)).T
        for layer in self.dense_layers:
            pred = layer(pred)
        return pred

if __name__=='__main__':
    folder = rf"E:\FYP excel\{dataset}_MFCC{n_mfcc}_delta={delta}_LSTM_layer={num_layer}_hidden={hidden_layer_size}_linears={linear_size}_p={p}"
    if not os.path.exists(folder):
       os.makedirs(folder)

    csv_path = rf"E:\FYP excel\{dataset}_MFCC{n_mfcc}_train=0.8.csv"
    # csv_path = rf"E:\FYP excel\IEMOCAP2_AlexNet_6.csv"
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
    start_t = np.array(df_filtered.iloc[:, 0])  # 0 for normal, 1 for 4096
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
    for i in range(len(s)-1):
        pad = np.zeros([length-dur[i],X.shape[1]])
        feature = np.concatenate([X[s[i]:s[i+1]],pad])
        features.append(feature)
        # print(y[s[i]])
        labels.append(y[s[i]])
    pad = np.zeros([length-dur[-1],X.shape[1]])
    pad[:, :] = np.nan
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

    dl_train = DataLoader(TensorDataset(X_train,y_train,dur_train), batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(TensorDataset(X_test,y_test,dur_test), batch_size=batch_size, shuffle=True)



    if __name__ =='__main__':
        model,min_loss,max_acc,model_list = load_model(folder)
        if model is None:
            model = LSTM(input_size=input_size, hidden_layer_size=hidden_layer_size, linear_size=linear_size, output_size=output_size, batch_size=batch_size,num_layer=num_layer).to(device)
        model.dense_layers[0] = nn.Dropout(p=dropout, inplace=False)
        model.dense_layers[3] = nn.Dropout(p=dropout, inplace=False)
        if test_mode:
            model = load_best_model(folder)
            model.eval()
        print(f'model={model}')
        model = model.to(device)

        metric_func = accuracy
        metric_name = "accuracy"
        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.to(device)
        def loss_func(y_pred,y_true):
            one_hot = torch.zeros((len(y_true),output_size)).to(device)
            for a,b in zip(one_hot,y_true):
                a[b]=1
            # print(y_pred.size(),one_hot.size())
            return loss_function(y_pred,one_hot)


        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

        train(model, dl_train,dl_test,loss_function=loss_function, metric_function=metric_func, optimizer=optimizer, epochs=10000, batch_size=batch_size,
              model_name='LSTM',df_path=folder+f'\\{os.path.split(folder)[-1]}.csv')

        #fr"D:\pycharmProject\FYP\deep learning\IEMOCAP1_LSTM_MFCC{n_mfcc}_delta{delta}.csv"
