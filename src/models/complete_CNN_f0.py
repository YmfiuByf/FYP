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
from preprocessing import load_model, load_best_model,voiced_index
from librosa_mfcc_feature_extraction import feature_IEMOCAP
from librosa_mfcc_SAVEE import librosa_SAVEE
from sklearn.metrics import accuracy_score
import pandas as pd
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
            dropout: float = 0.5,
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
        # Adaptive average pooling
        # print(x.shape)
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

class f0attCNN(nn.Module):
    def __init__(self, n_mfcc=66, kernel_length=[], hidden_layer=[], Cin=3, Cout=256, k=[],output_size = 6):
        super().__init__()
        self.num_kernel = len(kernel_length)
        self.num_layer = len(hidden_layer)
        self.Cout = Cout
        self.convs = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for KL,length in zip(kernel_length,k):
            self.convs.append(
                nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=[KL, n_mfcc], stride=max(KL // 3, 1)).to(
                    device))
            self.convs2.append(
                nn.Conv1d(in_channels=1, out_channels=Cout, kernel_size=KL, stride=max(KL//3,1)).to(
                    device))
            self.attentions.append(
                nn.Linear(Cout,length)
            )

        self.adPooling = nn.ModuleList()
        self.k = k
        if pooling == 'max':
            for length in k:
                self.adPooling.append(nn.AdaptiveMaxPool2d((length, 1)))
        elif pooling == 'average':
            for length in k:
                self.adPooling.append(nn.AdaptiveAvgPool2d((length, 1)))
        self.linears = nn.ModuleList()
        self.last_linear = nn.Linear(hidden_layer[-1], output_size)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.input_size = sum(k) * self.Cout
        for i in range(self.num_layer):
            if i == 0:
                input = self.input_size
            else:
                input = hidden_layer[i - 1]
            # print(hidden_layer[i],input)
            self.linears.append(nn.Linear(input, hidden_layer[i]).to(device))
        self.dropout = nn.Dropout(p=0.75, inplace=False)
        # print(self.convs,self.linears)

    def forward(self, x,f0):
        out = torch.tensor([]).to(device)
        for conv,conv2,attention, pool, length in zip(self.convs,self.convs2,self.attentions,self.adPooling, self.k):
            # print(conv)
            f = conv(x)
            att = conv2(f0)
            att = attention(att.transpose(-1,-2))
            att = torch.softmax(att,dim=-2).unsqueeze(1)
            f = torch.matmul(f.transpose(-1,-2),att)
            f = torch.flatten(f,start_dim=1)
            # f = pool(f).reshape(len(f), -1)
            out = torch.cat((out, f), dim=1)
        for linear in self.linears:
            out = self.dropout(out)
            out = linear(out)
            out = self.relu(out)
        out = self.last_linear(out)
        return out

class Conv_Block(nn.Module):
    def __init__(self,m=3,t=3):
        super(Conv_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=[t,m],stride=[t//2,m//2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=[t,m],stride=[1,1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=[t, m], stride=[t // 2, m // 2]),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=[t,m],stride=[1,1]),
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
            # print(x_.shape,f_.shape,conv)
            att = self.attention_f(f_.transpose(-1,-2))
            att = torch.softmax(att,dim=-2).unsqueeze(1)
            x_ = torch.matmul(x_.transpose(-1,-2),att).squeeze(-1)
            # x_ = torch.flatten(x_,start_dim=1)
            att = attention(x_.transpose(-1,-2))
            att = torch.softmax(att, dim=-2)
            x_ = torch.matmul(x_,att).squeeze(-1)
            out = torch.concat([out,x_],dim=-1)
        out = self.classifier(out)
        # print(out.shape)
        return out

# class CNN_attention(nn.Module):
#     def __init__(self, n_mfcc=15,output_size = 6):
#         super().__init__()
#         self.num_kernel = len(kernel_length)
#         self.num_layer = len(hidden_layer)
#         self.Cout = Cout
#         self.convs = nn.ModuleList()
#         for KL in kernel_length:
#             self.convs.append(
#                 nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=[KL, n_mfcc], stride=max(KL // 3, 1)).to(
#                     device))
#         self.adPooling = nn.ModuleList()
#         self.k = k
#         if pooling == 'max':
#             for length in k:
#                 self.adPooling.append(nn.AdaptiveMaxPool2d((length, 1)))
#         elif pooling == 'average':
#             for length in k:
#                 self.adPooling.append(nn.AdaptiveAvgPool2d((length, 1)))
#         self.linears = nn.ModuleList()
#         self.last_linear = nn.Linear(hidden_layer[-1], output_size)
#         self.relu = nn.ReLU(inplace=True)
#         self.tanh = nn.Tanh()
#         self.input_size = sum(k) * self.Cout
#         for i in range(self.num_layer):
#             if i == 0:
#                 input = self.input_size
#             else:
#                 input = hidden_layer[i - 1]
#             # print(hidden_layer[i],input)
#             self.linears.append(nn.Linear(input, hidden_layer[i]).to(device))
#         self.dropout = nn.Dropout(p=0.75, inplace=False)
#         # print(self.convs,self.linears)
#
#     def forward(self, x):
#         out = torch.tensor([]).to(device)
#         for conv, pool, length in zip(self.convs, self.adPooling, self.k):
#             # print(conv)
#             f = conv(x)
#             f = pool(f).reshape(len(f), -1)
#             out = torch.cat((out, f), dim=1)
#         for linear in self.linears:
#             out = self.dropout(out)
#             out = linear(out)
#             out = self.relu(out)
#         out = self.last_linear(out)
#         return out

# kernel_length=[5,10,15,20,25]
# hidden_layer=[4096,2048,1024]
def train_CNN(test_mode=False,n_mfcc=70,method='inception',
              dataset='SAVEE',folder=None,remove_silence=False,
              lr=1e-4,weight_decay=0.1,dropout =0.7,batch_size=16,m=[3,5],t=[3,5]):
    result_path = r"E:\FYP excel\Results"
    save_csv_path = r"E:\FYP excel\Results\result.csv"
    kernel_length=[1,2,4,8,16,32,64]
    k=[32,24,16,12,8,4,2]
    # kernel_length = [1,2,3,4,5]
    # k = [5,4,3,2,1]
    hidden_layer=[4096,4096]
    bg = 40
    min_length = bg
    ed = None
    max_length = 100000 if ed is None else ed
    # lr,weight_decay,dropout =  1e-4,   0.1,   0.7
    # # lr,weight_decay,dropout =  1e-6,   0.5,   0.75
    # # lr,weight_decay,dropout =  1e-7,   1,   0.85
    seed = 1

    test_mode = test_mode

    pooling = 'max'
    n_mfcc = n_mfcc
    # dataset = 'IEMOCAP1'
    # dataset = 'SAVEE'
    # data

    if not test_mode:
        remove_silence = False
    str = '_voiced' if remove_silence else ''
    if not test_mode:
        if method =='CNNv1':
            folder = rf"E:\FYP excel\{dataset}_{method}{str}_kernel={kernel_length}_linear={hidden_layer}_k={k}_seed{seed}_mfcc{n_mfcc}_{pooling}pooling"
    if not os.path.exists(folder):
        if test_mode:
            return
        os.makedirs(folder)

    csv_path = fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}_train=0.8.csv"
    csv_f0 = fr"E:\FYP excel\{dataset}_MFCC0_train=0.8.csv"
    if test_mode:
        csv_path = fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}_test=0.2.csv"
        csv_f0 = fr"E:\FYP excel\{dataset}_MFCC0_test=0.2.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        func_dict = {'IEMOCAP1':feature_IEMOCAP,'SAVEE':librosa_SAVEE}
        function = func_dict[dataset]

        # if not test_mode:
        #     for i in range(1,200):
        #         if os.path.exists(fr"E:\FYP excel\{dataset}_MFCC{i}_test=0.2.csv"):
        #             os.remove(fr"E:\FYP excel\{dataset}_MFCC{i}_test=0.2.csv")
        #         if os.path.exists(fr"E:\FYP excel\{dataset}_MFCC{i}_train=0.8.csv"):
        #             os.remove(fr"E:\FYP excel\{dataset}_MFCC{i}_train=0.8.csv")

        if dataset=='IEMOCAP1':
            df_train, df_test = feature_IEMOCAP(n_mfcc=n_mfcc)
        elif dataset=='SAVEE':
            df_train, df_test = librosa_SAVEE(n_mfcc=n_mfcc)
        if test_mode:
            df = df_test
            # del df_train
        else:
            df = df_train
            # del df_test
        df_train.to_csv(fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}_train=0.8.csv")
        df_test.to_csv(fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}_test=0.2.csv")
        if test_mode:
            df = pd.read_csv(fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}_test=0.2.csv")
        else:
            df = pd.read_csv(fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}_train=0.8.csv")
        # df.insert(loc=0, column='redundant', value=np.arange(len(df)))
        # print(f'df={df}')
    df_f0 = pd.read_csv(csv_f0)
    print(np.unique(df['label']))
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

    if 9 in np.array(df_f0['label']):
        print('remove by labels')
        df_f0 = df_f0[df_f0['label']!=0]
        df_f0 = df_f0[df_f0['label']!=3]
        df_f0 = df_f0[df_f0['label']!=6]
        df_f0 = df_f0[df_f0['label']!=7]
        df_f0['label'][df_f0['label'] == 8] = 3
        df_f0['label'][df_f0['label'] == 9] = 6

    f0 = np.array(df_f0.iloc[:,1])
    f0[np.where(np.isnan(f0))]=0.
    dict = {'IEMOCAP1':6,'SAVEE':7,'RAVDESS':8}
    output_size = dict[dataset]

    voiced,index = voiced_index(df_f0)

    num_samples = None#120000
    arr = df_filtered.to_numpy()
    del df,df_filtered
    print(f'arr={arr.shape}')
    X = arr[:,1:-1]
    print(X)
    y = arr[:,-1]
    if remove_silence:
        X = X[voiced]
        y = y[voiced]
        f0 = f0[voiced]
        assert len(X)==len(y)
        assert len(X)==len(index)
    print(np.unique(y))
    # X, _, y, __ = train_test_split(arr[:,1:-1].reshape((-1,n_mfcc,3),order='F'), arr[:,-1], train_size=num_samples,random_state=0)
    print(f'X={X.shape}')
    # X=arr[:num_samples,3:-1]
    # print(X.shape)
    # y=arr[:num_samples,-1]
    y=y.astype('int')
    length = len(X)
    if remove_silence:
        starts = index==0
    else:
        start_t = arr[0][0]
        starts = arr[:,0] == start_t
    # starts = index == 0

    s = np.arange(len(starts))[starts]
    dur = (s[1:]-s[:-1])
    dur = np.append(dur,len(starts)-s[-1])
    length = np.max(dur)+1
    print(f'min={min(dur)},max={max(dur)}')
    bg = np.where(np.unique(dur,return_counts=True)[0]>=bg)[0][0]
    if ed is not None:
        ed = np.where(np.unique(dur,return_counts=True)[0]>=ed)[0][0]
    print(np.unique(dur,return_counts=True)[1][bg:ed].sum()/len(dur),np.unique(dur,return_counts=True)[0][bg:ed])
    features,labels,f0s = [],[],[]
    dur_idx = np.full(len(dur),True)
    for i in range(len(s)-1):
        if dur[i]<min_length or dur[i]>max_length:
            dur_idx[i] = False
            continue
        pad = np.zeros([length-dur[i],X.shape[1]])
        pad_f0 = np.zeros(length-dur[i])
        X_seg = X[s[i]:s[i+1]]
        f0_seg = f0[s[i]:s[i+1]]
        # if remove_silence:
        #     print(voiced[s[i]:s[i+1]])
        #     X_seg = X_seg[voiced[s[i]:s[i+1]]]
        feature = np.concatenate([X_seg,pad])
        f0_seg = np.concatenate([f0_seg,pad_f0])
        features.append(feature)
        f0s.append(f0_seg)
        labels.append(y[s[i]])
    pad = np.zeros([length-dur[-1],X.shape[1]])
    pad_f0 = np.zeros(length-dur[-1])
    feature = np.concatenate([X[s[-1]:len(starts)],pad])
    features.append(feature)
    f0s.append(np.concatenate([f0[s[-1]:len(starts)],pad_f0]))
    labels.append(y[s[-1]])
    features = np.array(features,dtype=np.float32).reshape((len(features),length,n_mfcc,3),order='F')
    labels = np.array(labels,dtype=np.int8)
    f0s = np.array(f0s,dtype=np.float32)[:,None,:]

    dur = dur[dur_idx==True]
    print('feature obtained')

    X = features
    y = labels
    X =np.moveaxis(X,-1,1)
    print(X.shape)
    print(f'f0={f0s.shape}')

    y = y-1
    print(np.unique(y,return_counts=True))

    # labels = np.zeros([len(y),9],dtype=np.int32)
    # for (l,i) in zip(labels,y):
    #     l[i-1] = 1
    # y = labels
    tsz = len(X)-1 if test_mode else 0.2
    X_train, X_test, y_train, y_test,dur_train,dur_test,f0_train,f0_test = train_test_split(X, y, dur,f0s, test_size=tsz,random_state=seed)
    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor).to(device)
    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor).to(device)
    dur_train = torch.from_numpy(dur_train).to(device)
    dur_test = torch.from_numpy(dur_test).to(device)
    f0_train = torch.from_numpy(f0_train).to(device)
    f0_test = torch.from_numpy(f0_test).to(device)
    dl_train = DataLoader(TensorDataset(X_train,y_train,dur_train,f0_train), batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(TensorDataset(X_test,y_test,dur_test,f0_test), batch_size=batch_size, shuffle=True)

    print('loader built')

    def accuracy(y_pred,y_true):
        y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
        return accuracy_score(y_true.cpu(),y_pred_cls.cpu())


    def Train(model, dl_train,dl_test,loss_function, metric_function, optimizer, epochs=150, batch_size=32,model_name='CNN_dd',df_path=folder+f'\\{dataset}_{method}_MFCC{n_mfcc}.csv',min_loss=10000.,model_list=[],max_acc=0.):
        optimal = 0.
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
        if test_mode:
            model.eval()
            val_loss_sum = 0.
            val_metric_sum = 0.
            CM_val = torch.zeros((output_size, output_size)).int().to(device)
            epoch = 0
            y_preds = torch.tensor([]).to(device)
            y_true = torch.tensor([]).to(device)
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

                    y_preds = torch.cat((y_preds, torch.argmax(predictions, axis=1)))
                    y_true = torch.cat((y_true, labels))
                    assert y_preds.size() == y_true.size()
                    val_loss = loss_function(predictions, labels)
                    val_metric = metric_func(predictions, labels)

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
            display_dict = {'IEMOCAP1':['neu', 'sad', 'hap', 'fru', 'ang', 'exc'],'SAVEE':['anger', 'disgust', 'fear', 'happiness', 'neural', 'sad', 'surprise']
}
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=display_dict[dataset])
            disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,
                                          display_labels=display_dict[dataset])
            disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                                          display_labels=display_dict[dataset])
            # name = f'{dataset}_CNN{str}_kernel={kernel_length}_linear={hidden_layer}_k={k}_seed{seed}_mfcc{n_mfcc}_{pooling}pooling'
            print(f'folder is {folder}')
            name = folder[folder.index('excel')+6:]
            print(name)
            disp.plot()
            plt.title(folder)
            plt.savefig(os.path.join(result_path, f'{name}.png'))
            disp1.plot()
            plt.title(folder)
            plt.savefig(os.path.join(result_path, f'Precision_{name}.png'))
            disp2.plot()
            plt.title(folder)
            plt.savefig(os.path.join(result_path, f'Recall_{name}.png'))
            print(f'CM_val={CM_val}')
            result = np.array([name, val_loss_sum / val_step, val_metric_sum / val_step])
            result = np.concatenate([result, precision])
            result = np.concatenate([result, recall])
            result = result[None, :]
            if dataset == 'IEMOCAP1':
                result = np.concatenate([result, np.zeros((1, 2))], axis=1)
            # result = pd.DataFrame(result)
            if os.path.exists(save_csv_path):
                df = pd.read_csv(save_csv_path)
                df = df.iloc[:,1:]
                # if name in np.array(df.iloc[:,0]):
                # result = result[None,:]
                print(f'dataset is {dataset}')
                df.loc[len(df),:] = result
            else:
                df = pd.DataFrame(result)
            df.to_csv(save_csv_path)

            print(f'CM_val={CM_val}')
            # 3，记录日志-------------------------------------------------
            info = (epoch,val_loss_sum / val_step, val_metric_sum / val_step)
            # 打印epoch级别日志
            print(("\nEPOCH = %d" +", test_loss = %.3f, " + "test_" + metric_name + " = %.3f")
                  % info)
            return

        for epoch in range(len(dfhistory)+1,epochs+1):
            model.train()
            loss_sum = 0.0
            metric_sum = 0.0
            step = 1
            CM_train = torch.zeros((output_size,output_size)).int().to(device)
            CM_val = torch.zeros((output_size, output_size)).int().to(device)
            for step, (features,y_true,dur,f0s) in enumerate(dl_train,1):
                #print(seq.size(),labels.size())
                optimizer.zero_grad()
                # print(dur)
                if len(features)>1:
                    y_pred = torch.tensor([]).to(device)
                    for i in range(len(features)):
                        feature = features[i:i+1, :, :dur[i], :]
                        f0 = f0s[i:i+1, :, :dur[i]]
                        # print(feature.shape,f0.shape)
                        # print(dur[i])
                        # print(feature.size())
                        y_pred = torch.cat((y_pred,model(feature,f0)),dim=0)
                        # print(y_pred.size())
                else:
                    features = features[:,:,:dur,:]
                    f0 = f0s[:, :, :dur]
                    # print(f'fshape={features.size()}')
                    # features = torch.transpose(features,1,-1)
                    y_pred = model(features,f0)
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

            for val_step, (features,labels,dur,f0s) in enumerate(dl_test,1):
                with torch.no_grad():
                    if len(features) > 1:
                        y_pred = torch.tensor([]).to(device)
                        for i in range(len(features)):
                            feature = features[i:i + 1, :, :dur[i], :]
                            f0 = f0s[i:i + 1, :, :dur[i]]
                            # print(dur[i])
                            # print(feature.size())
                            y_pred = torch.cat((y_pred, model(feature,f0)), dim=0)
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
            dfhistory.loc[epoch - 1] = info + (np.nan,)*output_size
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
            if loss_sum / step < min_loss or int(val_metric_sum * 1000 / val_step)>max_acc:
                min_loss = loss_sum / step
                model_save_path = folder + f'\\{dataset}_CNN_MFCC{n_mfcc}_{val_metric_sum / val_step:.3f}_loss={min_loss:.3f}.pth'.replace(
                    '0.', '', 1)
                torch.save(model, model_save_path)
                model_list.append(model_save_path)
                if int(val_metric_sum * 1000 / val_step) >= max_acc:
                    max_acc = int(val_metric_sum * 1000 / val_step)
                    if model_save_path not in model_list:
                        torch.save(model,model_save_path)
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
            print(df_path)
        return


    # def __init__(self,input_sample, hidden_layer=[1024,512,128], kernel_size = [[3,3,16],[3,3,16],[3,3,16]], output_size=9):


    # def train(model, dl_train,dl_test,loss_function, metric, optimizer, epochs=150, batch_size=32,model_path=r'D:\pycharmProject\FYP\deep learning\CNN',df_path=r"D:\pycharmProject\FYP\deep learning\CNN"):
    # CNN = myCNN(X_train[0]).to(device)
    # print(CNN(X_train[0:16]).size())
    # CNN = torch.load(r"D:\pycharmProject\FYP\deep learning\CNN_dd.pth")

    CNN,min_loss,max_acc,model_list = load_model(folder)
    if CNN is None:
        if method == 'CNNv1':
            CNN = f0attCNN(n_mfcc=n_mfcc,kernel_length=kernel_length, hidden_layer=hidden_layer, Cin=3, Cout=256, k=k,output_size=output_size).to(device)
            CNN.dropout = nn.Dropout(p=dropout, inplace=False)
        elif method in ['Inception','inception']:
            CNN = SER_Inception(num_classes=output_size)
        elif 'attentionCNN' in method:
            CNN = CNN_attention(ms=m,ts=t,output_size = output_size)
    CNN = CNN.to(device)
    # CNN.classifier[0] = nn.Dropout(p=dropout, inplace=False)
    # CNN.classifier[3] = nn.Dropout(p=dropout, inplace=False)
    # CNN = torch.load(r"E:\FYP excel\IEMOCAP1_encoder_CNN\IEMOCAP1_CNN_MFCC30_402_loss=1.816.pth").to(device)
    if test_mode:
        CNN = load_best_model(folder)
        CNN.eval()
    print(f'CNN={CNN}')
    metric_func = accuracy
    metric_name = "accuracy"
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)


    optimizer = torch.optim.Adam(CNN.parameters(), lr=lr,weight_decay=weight_decay)

    Train(model=CNN,dl_train=dl_train,dl_test=dl_test,loss_function=loss_function,metric_function=metric_func,optimizer=optimizer,epochs=10000000000,min_loss=min_loss,model_list=model_list,max_acc=max_acc)

if __name__ =='__main__':
    # train_CNN(test_mode=True,folder=)
    # lr,weight_decay,dropout =  1e-4,   0.1,   0.7
    # lr,weight_decay,dropout =  1e-6,   0.5,   0.75
    lr,weight_decay,dropout =  1e-7,   1,   0.85
    method = 'inception'
    method = 'CNNv1'
    # method = 'attentionCNN'
    m,t = [3,5], [3,5]

    dataset = 'IEMOCAP1'
    dataset = 'SAVEE'
    n_mfcc = 66
    remove_silence = False
    batch_size = 4
    train_CNN(test_mode=False,method=method,n_mfcc=n_mfcc,dataset=dataset,
              folder=rf"E:\FYP excel\{dataset}_{method}_mfcc{n_mfcc}",
              remove_silence=remove_silence,lr=lr,weight_decay=weight_decay,dropout=dropout,batch_size=batch_size,
              m=m,t=t)
    pass

