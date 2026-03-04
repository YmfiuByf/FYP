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
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import  Tensor
from torchvision.models import inception
pooling = 'max'
class SER_Inception(nn.Module):
    def __init__(
            self,
            num_classes: int = 6,
            transform_input: bool = False,
            dropout: float = 0.7,
            attention=False
    ):
        super(SER_Inception, self).__init__()
        inception_blocks = [inception.BasicConv2d, inception.InceptionA, inception.InceptionB, inception.InceptionC, inception.InceptionD, inception.InceptionE, inception.InceptionAux]
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]
        self.attention = attention
        self.attention_q = nn.Linear(2048,2048)
        self.attention_k = nn.Linear(2048,2048)
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=1)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, num_classes)


    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tensor:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17

        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        if self.attention:
            # Attention pooling
            x = torch.flatten(x, 2)
            q = self.attention_q(x.transpose(1,2).squeeze())
            k = self.attention_k(x.transpose(1,2).squeeze())
            att = torch.matmul(q, k.transpose(-1,-2))
            att = torch.softmax(att,dim=-2)
            # print(q.shape,k.shape,att.shape,x.shape)
            x = torch.matmul(x,att)
            x = torch.sum(x,dim=-1,keepdim=False)
        else:
            # Adaptive average pooling
            x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x

    def forward(self, x: Tensor) -> inception.InceptionOutputs:
        x = self._transform_input(x)
        x = self._forward(x)
        return x

class Conv_Block(nn.Module):
    def __init__(self,m=3,t=3):
        super(Conv_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=[m,t],stride=[m//2,t//2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=[m,t],stride=[1,1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=[m, t], stride=[m // 2, t // 2]),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=[m,t],stride=[1,1]),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=t, stride=t//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=t, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=t, stride=t//2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=t, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5,inplace=False),
        )
    def forward(self,x,f):
        return self.block(x),self.block2(f)

class CNN_attention(nn.Module):
    def __init__(self, ms=[3,5],ts=[3,5],output_size = 6):
        super().__init__()
        self.convs = nn.ModuleList()
        for m in ms:
            for t in ts:
                self.convs.append(Conv_Block(m=m,t=t))
        # self.convs.append(Conv_Block(3,3))
        # self.convs.append(Conv_Block(5,5))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(len(self.convs)*256,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,output_size)
        )
        self.attentions = nn.ModuleList()
        for _ in range(len(self.convs)):
            self.attentions.append(nn.Linear(256,1))
        self.attention_f = nn.Linear(256,1)

        # print(self.convs,self.linears)
    def forward(self, x,f):
        out = torch.tensor([]).to(device)
        for conv,attention in zip(self.convs,self.attentions):
            x_, f_ = conv(x,f)
            att = self.attention_f(f_.transpose(-1,-2))
            att = torch.softmax(att,dim=-2).unsqueeze(1)
            x_ = torch.matmul(x_,att).squeeze(-1)
            # x_ = torch.flatten(x_,start_dim=1)
            att = attention(x_.transpose(-1,-2))
            att = torch.softmax(att, dim=-2)
            x_ = torch.matmul(x_,att).squeeze(-1)
            out = torch.concat([out,x_],dim=-1)
        return out

import pandas as pd

CUDA_LAUNCH_BLOCKING=1
torch.cuda.empty_cache()

# pooling = 'max'
lr,weight_decay,dropout = 1e-4, 0.5, 0.7

seed = 0
m,t=[3,5], [3,5]
method = 'inception'
method = f'attentionCNN_m={m}_t={t}'
# method = 'AlexNet'
test_mode = False
n_mfcc = 32
length = 32
step = 4
batch_size = 32
dataset = 'IEMOCAP1'
dataset = 'SAVEE'
attention = False
folder = rf"E:\FYP excel\{dataset}_{method}_{attention}_mfcc{n_mfcc}"
# folder = rf"E:\FYP excel\IEMOCAP1_AlexNetNew768_voiced_mfcc66"
# folder = rf"E:\FYP excel\SAVEE_{method}_new_mfcc66"
# folder = r"E:\FYP excel\IEMOCAP1_AlexNetNew768_voiced_mfcc66"
not_allow_to_load= False

if not_allow_to_load:
    if os.path.exists(folder):
        os.remove(folder)
if not os.path.exists(folder):
   os.makedirs(folder)


csv_path = fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}.csv"
csv_path1 = fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}_train=0.8.csv"
csv_path2 = fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}_test=0.2.csv"
csv_f01 = fr"E:\FYP excel\{dataset}_MFCC0_train=0.8.csv"
csv_f02 = fr"E:\FYP excel\{dataset}_MFCC0_test=0.2.csv"


def voiced_index(df):
    print(len(df))
    time = np.array(df.iloc[:,0])
    print(f'time={time},{len(time)}')
    starts = time== 0
    starts = np.arange(len(starts))[starts]
    total_voiced = np.array([])
    index = np.array([])
    for i in range(len(starts)):
        if i == len(starts)-1:
            ed = len(time)
        else:
            ed = starts[i+1]
        bg = starts[i]
        voiced_part = np.array(df['flag'][bg:ed])
        total_voiced = np.concatenate([total_voiced,voiced_part])
        index = np.concatenate([index,np.arange(len(voiced_part))])
        # voiced_part[1:] += voiced_part[:-1]
        # voiced_part[1:] += voiced_part[:-1]
        # voiced_part[1:] += voiced_part[:-1]
    # print(np.unique(total_voiced!=0,return_counts=True))
    print(len(total_voiced))
    total_voiced = np.arange(len(total_voiced))[total_voiced!=0]
    print(len(total_voiced))
    return total_voiced, index

output_size = 6 if dataset=='IEMOCAP1' else 7
if dataset == 'RAVDESS':
    output_size = 8

def load_dataset(csv_path,csv_path_f0):
    df = pd.read_csv(csv_path)
    df_f0 = pd.read_csv(csv_path_f0)
    print(f'lengths={len(df)},{len(df_f0)}')
    assert len(df)==len(df_f0)
    voiced,index = voiced_index(df_f0)
    # print(df.iloc[voiced,:])
    # df = df.iloc[voiced,:]

    print('label is',np.unique(df['label']),9 in np.array(df['label']))

    if 9 in np.array(df['label']):
        print('remove by labels')
        df_filtered = df[df['label']!=0]
        df_filtered = df_filtered[df_filtered['label']!=3]
        df_filtered = df_filtered[df_filtered['label']!=6]
        df_filtered = df_filtered[df_filtered['label']!=7]
        df_filtered['label'][df_filtered['label'] == 8] = 3
        df_filtered['label'][df_filtered['label'] == 9] = 6
    else:
        df_filtered = df
    num_samples = None#120000
    arr = df_filtered.to_numpy()
    del df,df_filtered
    print(f'arr={arr.shape}')
    X = arr[:,1:-1]
    y = arr[:,-1]
    print(np.unique(y))
    print(f'X={X.shape}')
    y=y.astype('int')
    # y -= 1
    length = len(X)
    starts = index == 0
    # start_t = arr[0][0]
    # starts = arr[:,0] == start_t
    return X, y, starts

CNN,min_loss,max_acc,model_list = load_model(folder)
if CNN is None or not_allow_to_load:
    if method in ['AlexNet']:
        CNN = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        # CNN = torch.load(r"E:\FYP excel\IEMOCAP1_MFCC64_AlexNet\IEMOCAP1_AlexNet_MFCC64_652.pth")
        conv = CNN.features[0]
        CNN.features[0] = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(1, 1))
        CNN.classifier[0] = nn.Dropout(p=dropout, inplace=False)
        CNN.classifier[3] = nn.Dropout(p=dropout, inplace=False)
        CNN.classifier[4] = nn.Linear(4096, 768).to(device)
        CNN.classifier[6] = nn.Linear(768,output_size).to(device)
        CNN = CNN.to(device)
        for param, conv_param in zip(CNN.features[0].parameters(), conv.parameters()):
            param = conv_param
    elif method in ['inception']:
        CNN = SER_Inception(num_classes=output_size,attention=attention)
    elif 'attentionCNN' in method:
        print(f'method={method}')
        CNN = CNN_attention(ms=m,ts=t)
    CNN = CNN.to(device)
if method in ['AlexNet']:
    CNN.classifier[0] = nn.Dropout(p=dropout, inplace=False)
    CNN.classifier[3] = nn.Dropout(p=dropout, inplace=False)


X_train,y_train,starts_train = load_dataset(csv_path1,csv_f01)
X_test,y_test,starts_test = load_dataset(csv_path2,csv_f02)

# Fixed Length Feature
def get_feature(X,y,starts,l=length):
    cnn_features = []
    labels = []
    i,j = 0,l
    length = len(X)
    # X = X[:100]
    # y = y[:100]
    while j<length:
        if starts[j]:
            i = j
            j = i+l
            continue
        feature = X[i:j,:].reshape((l, n_mfcc, 3), order='F')
        cnn_features.append(feature)
        labels.append(y[i])
        i+= step
        j = i+l
    features = np.array(cnn_features,dtype=np.float32)
    print(features.shape)
    labels = np.array(labels,dtype=np.int8)
    X = features
    y = labels
    X = np.moveaxis(X, -1, 1)
    print(X.shape)
    y = y - 1
    print(np.unique(y, return_counts=True))
    return X, y

def get_feature_train_val(X,y,starts,l=length,val_size=0.1):
    global seed
    cnn_features,features_val = [],[]
    labels,labels_val = [],[]
    i,j = 0,l
    length = len(X)
    flag = 'train'
    # X = X[:100]
    # y = y[:100]
    while j<length:
        if starts[j]:
            i = j
            j = i+l
            np.random.seed(i+seed)
            if np.random.rand()>val_size:
                flag='train'
            else:
                flag='val'
            continue
        feature = X[i:j,:].reshape((l, n_mfcc, 3), order='F')
        if flag=='train':
            cnn_features.append(feature)
            labels.append(y[i])
        elif flag=='val':
            features_val.append(feature)
            labels_val.append(y[i])
        i+= step
        j = i+l
    features = np.array(cnn_features,dtype=np.float32)
    features_val = np.array(features_val,dtype=np.float32)
    print(features.shape)
    labels = np.array(labels,dtype=np.int8)
    X = features
    y = labels
    X = np.moveaxis(X, -1, 1)
    y = y - 1
    print(np.unique(y, return_counts=True))
    X_val = features_val
    y_val = np.array(labels_val,dtype=np.int8)
    X_val = np.moveaxis(X_val, -1, 1)
    y_val = y_val - 1
    print(np.unique(y_val, return_counts=True))
    return X, y, X_val,y_val
if test_mode:
    X_test, y_test = get_feature(X_test, y_test, starts_test)
    X_train,y_train = get_feature(X_train,y_train,starts_train)
    X_train = X_train[:160]
    y_train = y_train[:160]

else:
    X_train, y_train,X_test,y_test = get_feature_train_val(X_train, y_train, starts_train)
    while (len(np.unique(y_test)) < output_size):
        seed += 1
        X_train, y_train, starts_train = load_dataset(csv_path1, csv_f01)
        X_train, y_train,X_test,y_test = get_feature_train_val(X_train, y_train, starts_train)
        print(f'seed={seed}')
# X_test, y_test = get_feature(X_test,y_test,starts_test)

print('feature obtained')


# labels = np.zeros([len(y),9],dtype=np.int32)
# for (l,i) in zip(labels,y):
#     l[i-1] = 1
# y = labels

# X_train, X_test, y_train, y_test,dur_train,dur_test = train_test_split(X, y, dur, test_size=0.2,random_state=0)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
X_train = torch.from_numpy(X_train).to(device)
y_train = torch.from_numpy(y_train).type(torch.LongTensor).to(device)
X_test = torch.from_numpy(X_test).to(device)
y_test = torch.from_numpy(y_test).type(torch.LongTensor).to(device)
# dur_train = torch.from_numpy(dur_train).to(device)
# dur_test = torch.from_numpy(dur_test).to(device)
# dl_train = DataLoader(TensorDataset(X_train,y_train,dur_train), batch_size=16, shuffle=True)
# dl_test = DataLoader(TensorDataset(X_test,y_test,dur_test), batch_size=16, shuffle=True)

# del X,y
dl_train = DataLoader(TensorDataset(X_train,y_train), batch_size=batch_size, shuffle=True)
dl_test = DataLoader(TensorDataset(X_test,y_test), batch_size=batch_size, shuffle=True)
print('loader built')


# CNN = torch.load(r"E:\FYP excel\IEMOCAP1_encoder_CNN\IEMOCAP1_CNN_MFCC30_402_loss=1.816.pth").to(device)
print(f'CNN={CNN}')

from sklearn.metrics import accuracy_score

def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu(),y_pred_cls.cpu())

def Train(model, dl_train,dl_test,loss_function, metric_function, optimizer, epochs=150, batch_size=32,model_name='CNN_dd',df_path=folder+f'\\{dataset}_{method}_{attention}_MFCC{n_mfcc}.csv'):
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
        if dataset == 'SAVEE':
            dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name,'anger', 'disgust', 'fear', 'happiness', 'neural', 'sad', 'surprise'])
        elif dataset =='IEMOCAP1':
            dfhistory = pd.DataFrame(
                columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name, 'neu', 'sad', 'hap', 'fru',
                         'ang', 'exc'])  # ['anger', 'disgust', 'fear', 'happiness', 'neural', 'sad', 'surprise']
        elif dataset=='RAVDESS':
            dfhistory = pd.DataFrame(
                columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name, 'neu', 'calm', 'hap', 'sad', 'ang', 'fear', 'disgust','sur'])
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

    if test_mode:
        model.eval()
        epoch = 0
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1
        CM_val = torch.zeros((output_size, output_size)).int().to(device)
        y_preds = torch.tensor([]).to(device)
        y_true = torch.tensor([]).to(device)
        for val_step, (features,labels) in enumerate(dl_test,1):
            with torch.no_grad():
                # if len(features) > 1:
                #     y_pred = torch.tensor([]).to(device)
                #     for i in range(len(features)):
                #         feature = features[i:i + 1, :, :dur[i], :]
                #         # print(dur[i])
                #         # print(feature.size())
                #         y_pred = torch.cat((y_pred, model(feature)), dim=0)
                #         # print(y_pred.size())
                #     predictions=y_pred
                # else:
                #     features = features[:, :, :dur, :]
                #     # print(f'fshape={features.size()}')
                #     # features = torch.transpose(features,1,-1)
                #     predictions = model(features)

                predictions = model(features)
                val_loss = loss_function(predictions, labels)
                val_metric = metric_func(predictions, labels)

            y_preds = torch.cat((y_preds, torch.argmax(predictions, axis=1)))
            y_true = torch.cat((y_true, labels))
            val_loss_sum += val_loss.item()
            val_metric_sum += val_metric.item()

            prediction = torch.argmax(predictions,dim=1)
            tmp = torch.unique(output_size * labels + prediction, return_counts=True)
            tmp0 = torch.div(tmp[0], output_size, rounding_mode='trunc')
            tmp1 = tmp[0] % output_size
            CM_val[tmp0, tmp1] += tmp[1]

        y_true = y_true.detach().cpu().numpy()
        y_preds = y_preds.detach().cpu().numpy()
        cm = confusion_matrix(y_true, y_preds, labels=list(range(output_size)))
        diag = np.diagonal(cm)
        precision = diag / np.sum(cm, axis=0)
        recall = diag / np.sum(cm, axis=1)
        cm1 = cm / np.sum(cm, axis=0)
        cm2 = cm / np.sum(cm, axis=1)
        display_dict = {'IEMOCAP1': ['neu', 'sad', 'hap', 'fru', 'ang', 'exc'],
                        'SAVEE': ['anger', 'disgust', 'fear', 'happiness', 'neural', 'sad', 'surprise']
                        }
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=display_dict[dataset])
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,
                                       display_labels=display_dict[dataset])
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                                       display_labels=display_dict[dataset])
        # name = f'{dataset}_CNN{str}_kernel={kernel_length}_linear={hidden_layer}_k={k}_seed{seed}_mfcc{n_mfcc}_{pooling}pooling'
        print(f'folder is {folder}')
        name = folder[folder.index('excel') + 6:]
        print(name)
        disp.plot()
        plt.title(folder)
        # plt.savefig(f'{folder}.png')
        disp1.plot()
        plt.title(val_metric_sum / val_step)
        plt.savefig(f'{folder}_precision.png')
        disp2.plot()
        plt.title(folder)
        # plt.savefig(f'Recall_{folder}.png')
        print(f'CM_val={CM_val}')
        # 3，记录日志-------------------------------------------------
        info = (epoch, val_loss_sum / val_step, val_metric_sum / val_step)

        # 打印epoch级别日志
        print(("\nEPOCH = %d" +  " val_loss = %.3f, " + "val_" + metric_name + " = %.3f")
              % info)
        return

    for epoch in range(len(dfhistory)+1,epochs+1):
        model.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        CM_train = torch.zeros((output_size,output_size)).int().to(device)
        CM_val = torch.zeros((output_size, output_size)).int().to(device)
        for step, (features,y_true) in enumerate(dl_train,1):
            #print(seq.size(),labels.size())
            optimizer.zero_grad()
            # print(dur)
            # #variable length
            # if len(features)>1:
            #     y_pred = torch.tensor([]).to(device)
            #     for i in range(len(features)):
            #         feature = features[i:i+1, :, :dur[i], :]
            #         # print(dur[i])
            #         # print(feature.size())
            #         y_pred = torch.cat((y_pred,model(feature)),dim=0)
            #         # print(y_pred.size())
            # else:
            #     features = features[:,:,:dur,:]
            #     # print(f'fshape={features.size()}')
            #     # features = torch.transpose(features,1,-1)
            #     y_pred = model(features)
            # #variable length

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

        for val_step, (features,labels) in enumerate(dl_test,1):
            with torch.no_grad():
                # if len(features) > 1:
                #     y_pred = torch.tensor([]).to(device)
                #     for i in range(len(features)):
                #         feature = features[i:i + 1, :, :dur[i], :]
                #         # print(dur[i])
                #         # print(feature.size())
                #         y_pred = torch.cat((y_pred, model(feature)), dim=0)
                #         # print(y_pred.size())
                #     predictions=y_pred
                # else:
                #     features = features[:, :, :dur, :]
                #     # print(f'fshape={features.size()}')
                #     # features = torch.transpose(features,1,-1)
                #     predictions = model(features)

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
        padding = (np.nan,)*output_size
        dfhistory.loc[epoch - 1] = info + padding
        if epoch%10 == 0:
            dfhistory.iloc[epoch-1-output_size:epoch-1, -output_size:] = CM_val.cpu().numpy()

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
            model_save_path = folder+f'\\{dataset}_{method}_MFCC{n_mfcc}_{val_metric_sum / val_step:.3f}_loss={min_loss:.3f}.pth'.replace('0.','',1)
            torch.save(model,model_save_path)
            model_list.append(model_save_path)
            if int(val_metric_sum*1000 / val_step) >=max_acc:
                max_acc = int(val_metric_sum*1000 / val_step)
            if len(model_list)>4:
                delete_path = model_list[0]
                id = delete_path.index('loss')
                acc = int(delete_path[id-4:id-1])
                if acc>= max_acc:
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


metric_func = accuracy
metric_name = "accuracy"
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
def loss_func(y_pred,y_true):
    one_hot = torch.zeros((len(y_true),output_size)).to(device)
    for a,b in zip(one_hot,y_true):
        a[b]=1
    return loss_function(y_pred,one_hot)


optimizer = torch.optim.Adam(CNN.parameters(), lr=lr,weight_decay=weight_decay)

Train(model=CNN,dl_train=dl_train,dl_test=dl_test,loss_function=loss_function,metric_function=metric_func,optimizer=optimizer,epochs=10000000000)



