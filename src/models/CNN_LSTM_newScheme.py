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
torch.cuda.empty_cache()
pooling = 'max'


class CNN_RNN(nn.Module):
    def __init__(self, n_mfcc=66, convs=[[64,11,1,0],[192,5,1,2],[384,3,1,1],[256,3,1,1],[256,3,1,1]],Cin=3, Cout=1024, output_size = 6,p=2):
        super().__init__()
        self.p=p
        self.features = nn.Sequential()
        size = n_mfcc
        for i in range(len(convs)):
            in_channel = convs[i-1][0]
            out_channel = convs[i][0]
            kernel_size = convs[i][1]
            stride = convs[i][2]
            padding = convs[i][3]
            size = int((size+1*padding-kernel_size)/stride)+1
            if i==0:
                in_channel = Cin
            self.features.append(nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding))
            self.features.append(nn.ReLU(inplace=True))
            if i in [0,1,4]:
                self.features.append(nn.MaxPool2d(3,2,0,1))
                size = int((size-3)/2+1)
        size = 6

        self.avgpool = nn.AdaptiveAvgPool1d((6,6))
        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Dropout(p=0.8, inplace=False))
        self.classifier.append(nn.Linear(in_features=1024, out_features=1024, bias=True))
        self.classifier.append(nn.ReLU(inplace=True))
        self.classifier.append(nn.Dropout(p=0.8, inplace=False))
        self.classifier.append(nn.Linear(in_features=1024, out_features=1024, bias=True))
        self.classifier.append(nn.ReLU(inplace=True))
        self.classifier.append(nn.Linear(in_features=1024, out_features=output_size, bias=True))

        self.hidden_layer_size = 1024
        self.lstm = nn.LSTM(size*convs[-1][0], self.hidden_layer_size,2,batch_first=True).to(device)



    def forward(self, x):
        # batch_size = len(x)
        # self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size),
        #                     torch.zeros(1, batch_size, self.hidden_layer_size))
        for layer in self.features:
            x = layer(x)
        # print(x.size())
        x = x.transpose(-1,-2)
        x = x.reshape(len(x),-1,x.size(-1))
        x = x.transpose(-1, -2)
        # print(x.size())
        # for layer in self.classifier:
        #     x = layer(x)
        x, (h_0,c_0) = self.lstm(x)
        # print(x.size())
        N = x.size(1)
        x = torch.norm(x, dim=1,p=self.p)*N**(-1/self.p)
        # x = x.transpose(0, 1)
        # x = x.reshape(len(x),-1)
        for layer in self.classifier:
            x = layer(x)

        return x


# kernel_length=[5,10,15,20,25]
# hidden_layer=[4096,2048,1024]
def train_CNNRNN(test_mode=False,n_mfcc=66,convs=[[64, 11, 1, 0], [192, 5, 1, 2], [384, 3, 1, 1], [256, 3, 1, 1], [256, 3, 1, 1]],dataset='SAVEE',folder=None,remove_silence=False):
    result_path = r"E:\FYP excel\Results"
    save_csv_path = r"E:\FYP excel\Results\result.csv"

    # n_mfcc=66
    # convs=[[64, 11, 1, 0], [192, 5, 1, 2], [384, 3, 1, 1], [256, 3, 1, 1], [256, 3, 1, 1]]
    Cin=3
    Cout=1024


    bg = 40
    min_length = bg
    ed = None
    max_length = 100000 if ed is None else ed
    # lr,weight_decay,dropout =  1e-4,   0.1,   0.7
    lr,weight_decay,dropout =  1e-5,   0.5,   0.75
    # lr,weight_decay,dropout =  1e-7,   1,   0.85
    seed = 1

    pooling = 'max'
    n_mfcc = n_mfcc
    dataset = 'IEMOCAP1'
    # dataset = 'SAVEE'
    p=2
    # data
    print(f'is test mode={test_mode}')
    remove_silence = False
    if test_mode:
        remove_silence = False
    str = '_voiced' if remove_silence else ''
    if not test_mode:
        folder = rf"E:\FYP excel\{dataset}_newCNNLSTM{str}_convs={convs}_seed{seed}_mfcc{n_mfcc}_{pooling}pooling"
    if not os.path.exists(folder):
        if test_mode:
            return
        os.makedirs(folder)


    csv_path = r"C:\Users\DELL\Desktop\ComParE_2016+label.csv"
    csv_path = r"C:\Users\DELL\Desktop\IEMOCAP_MFCC14+F0.csv"
    csv_path = r"C:\Users\DELL\Desktop\IEMOCAP_MFCC14_F0_delta_delta.csv"
    csv_path = fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}.csv"
    csv_path = fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}_train=0.8.csv"
    csv_f0 = fr"E:\FYP excel\{dataset}_MFCC0_train=0.8.csv"
    if test_mode:
        csv_path = fr"E:\FYP excel\{dataset}_MFCC{n_mfcc}_test=0.2.csv"
        csv_f0 = fr"E:\FYP excel\{dataset}_MFCC0_test=0.2.csv"
    # csv_path = r"C:\Users\DELL\Desktop\SAVEE+MFCC14+F0.csv"
    # csv_path = r"C:\Users\DELL\Desktop\IEMOCAP12345_ComParE_2016.csv"
    # csv_path = r"C:\Users\DELL\Desktop\SAVEE_ComParE_2016.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        func_dict = {'IEMOCAP1':feature_IEMOCAP,'SAVEE':librosa_SAVEE}
        function = func_dict[dataset]
        if not test_mode:
            for i in range(1,200):
                if os.path.exists(fr"E:\FYP excel\{dataset}_MFCC{i}_test=0.2.csv"):
                    os.remove(fr"E:\FYP excel\{dataset}_MFCC{i}_test=0.2.csv")
                if os.path.exists(fr"E:\FYP excel\{dataset}_MFCC{i}_train=0.8.csv"):
                    os.remove(fr"E:\FYP excel\{dataset}_MFCC{i}_train=0.8.csv")
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
    features,labels = [],[]
    dur_idx = np.full(len(dur),True)
    for i in range(len(s)-1):
        if dur[i]<min_length or dur[i]>max_length:
            dur_idx[i] = False
            continue
        pad = np.zeros([length-dur[i],X.shape[1]])
        X_seg = X[s[i]:s[i+1]]
        # if remove_silence:
        #     print(voiced[s[i]:s[i+1]])
        #     X_seg = X_seg[voiced[s[i]:s[i+1]]]
        feature = np.concatenate([X_seg,pad])
        features.append(feature)
        labels.append(y[s[i]])
    pad = np.zeros([length-dur[-1],X.shape[1]])
    pad[:, :] = np.nan
    feature = np.concatenate([X[s[-1]:len(starts)],pad])
    features.append(feature)
    labels.append(y[s[-1]])
    features = np.array(features,dtype=np.float32).reshape((len(features),length,n_mfcc,3),order='F')
    labels = np.array(labels,dtype=np.int8)

    dur = dur[dur_idx==True]
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
    tsz = len(X)-1 if test_mode else 0.2
    X_train, X_test, y_train, y_test,dur_train,dur_test = train_test_split(X, y, dur, test_size=tsz,random_state=seed)
    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor).to(device)
    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor).to(device)
    dur_train = torch.from_numpy(dur_train).to(device)
    dur_test = torch.from_numpy(dur_test).to(device)
    dl_train = DataLoader(TensorDataset(X_train,y_train,dur_train), batch_size=32, shuffle=True)
    dl_test = DataLoader(TensorDataset(X_test,y_test,dur_test), batch_size=32, shuffle=True)

    print('loader built')


    def accuracy(y_pred,y_true):
        y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
        return accuracy_score(y_true.cpu(),y_pred_cls.cpu())

    def Train(model, dl_train,dl_test,loss_function, metric_function, optimizer, epochs=150, batch_size=32,model_name='CNN_dd',df_path=folder+f'\\{dataset}_CNN_MFCC{n_mfcc}.csv',min_loss=10000.,model_list=[],max_acc=0.):
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
                    # print(predictions.size(),labels.size())
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
            display_dict = {'IEMOCAP1':['neu', 'sad', 'hap', 'fru', 'ang', 'exc'],
                            'SAVEE':['anger', 'disgust', 'fear', 'happiness', 'neural', 'sad', 'surprise']}
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=display_dict[dataset])
            disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,
                                          display_labels=display_dict[dataset])
            disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                                          display_labels=display_dict[dataset])
            name = f'{dataset}_CNNRNN{str}_convs={convs}_seed{seed}_mfcc{n_mfcc}_{pooling}pooling'
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
            # result = pd.DataFrame(result)
            if os.path.exists(save_csv_path):
                df = pd.read_csv(save_csv_path)
                df = df.iloc[:,1:]
                # if name in np.array(df.iloc[:,0]):
                df.loc[len(df),:] = result[None,:]
            else:
                df = pd.DataFrame(result[None,:])
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
        CNN = CNN_RNN(n_mfcc=n_mfcc, convs=convs,Cin=Cin, Cout=Cout, output_size = output_size,p=p).to(device)
    CNN.classifier[0] = nn.Dropout(p=dropout, inplace=False)
    CNN.classifier[3] = nn.Dropout(p=dropout, inplace=False)
    CNN.dropout = nn.Dropout(p=dropout, inplace=False)
    # CNN = torch.load(r"E:\FYP excel\IEMOCAP1_encoder_CNN\IEMOCAP1_CNN_MFCC30_402_loss=1.816.pth").to(device)
    if test_mode:
        CNN = load_best_model(folder)
        CNN.eval()
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


    optimizer = torch.optim.Adam(CNN.parameters(), lr=lr,weight_decay=weight_decay)

    Train(model=CNN,dl_train=dl_train,dl_test=dl_test,loss_function=loss_function,metric_function=metric_func,optimizer=optimizer,epochs=10000000000,min_loss=min_loss,model_list=model_list,max_acc=max_acc)

if __name__ =='__main__':
    train_CNNRNN(test_mode=False,n_mfcc=66,
                 convs=[[64, 11, 1, 0], [192, 5, 1, 2], [384, 3, 1, 1], [256, 3, 1, 1], [256, 3, 1, 1]],
                 dataset='SAVEE',remove_silence=False)

