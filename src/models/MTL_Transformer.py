import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor
from torch.utils.data import DataLoader, TensorDataset
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from preprocessing import *
import os
from sklearn.metrics import accuracy_score
torch.cuda.empty_cache()

def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu(),y_pred_cls.cpu())

clf = 'MLT_Transformer'
lr,weight_decay,dropout =  1e-4,   0.1,   0.7
# lr,weight_decay,dropout =  1e-5,   0.5,   0.75
# lr,weight_decay,dropout =  1e-7,   1,   0.85
w1,w2 = 1, 0
batch_size = 1
k=16
test_mode = False
dataset = 'IEMOCAP1'
# dataset = 'SAVEE'
folder = rf"E:\FYP excel\{dataset}_{clf}"
if not os.path.exists(folder):
   os.makedirs(folder)
output_size = 6 if dataset=='IEMOCAP1' else 7

csv_path = fr"E:\FYP excel\{dataset}_Transformer_data.csv"
df = pd.read_csv(csv_path)
y = np.array(df.columns[1:])
for i in range(len(y)):
    y[i] = np.int8(y[i][0])
y = y.astype(np.int8)
X = np.array(df.iloc[:,1:].T,dtype=np.float32)
dim = X[:,:3]/5.
X = X[:,3:]
# X[np.where(np.isnan(X))]=0 # pad with zeros
del df
device = 'cuda'
X_train, X_test, y_train, y_test,dim_train,dim_test = train_test_split(X, y, dim, test_size=0.2, random_state=1)

def remove_nan(a):
    x, y = np.where(np.isnan(a))
    cur = x[0]
    list = []
    list.append(a[cur, :y[x[0]]])
    for i,j in zip(x,y):
        if i <=cur:
            continue
        cur = i
        list.append(a[i,:j])
    return list

X_train = remove_nan(X_train)
X_test = remove_nan(X_test)
print(f"X={X_train[0]},dim={dim[0]},y={y[0]}")

# X_train = torch.from_numpy(X_train).to(device)
# y_train = torch.from_numpy(y_train).type(torch.LongTensor).to(device)
# X_test = torch.from_numpy(X_test).to(device)
# y_test = torch.from_numpy(y_test).type(torch.LongTensor).to(device)
# dim_train = torch.from_numpy(dim_train).to(device)
# dim_test = torch.from_numpy(dim_test).to(device)

# dl_train = DataLoader(TensorDataset(X_train,y_train,dim_train), batch_size=batch_size, shuffle=True)
# dl_test = DataLoader(TensorDataset(X_test,y_test,dim_test), batch_size=batch_size, shuffle=True)


class myReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.ModuleList()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(4,1024))
        self.classifier.append(nn.Dropout(p=dropout,inplace=False))
        self.classifier.append(nn.Linear(4096, 4096))
        self.classifier.append(nn.ReLU(inplace=True))
        self.classifier.append(nn.Dropout(p=dropout,inplace=False))
        self.classifier.append(nn.Linear(4096, output_size))


    def forward(self, features, **kwargs):
        x = features
        x = x.unsqueeze(1)
        x = self.pool(x)
        x = x.squeeze(1)
        x = x.reshape(len(x),-1)
        for layer in self.classifier:
            x = layer(x)
        return x

class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()
        self.classifier2 = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(4096, output_size)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(4, 1024))

    def forward(
            self,
            input_values,
    ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        #         print(hidden_states.size(),outputs[1].size(),outputs)#outputs[2].size(),outputs[3].size())
        hidden_states = torch.mean(hidden_states, dim=1)
        #         print(hidden_states.size())
        logits = self.classifier(hidden_states)

        x = outputs[0]
        x = x.unsqueeze(1)
        x = self.pool(x)
        x = x.squeeze(1)
        x = x.reshape(len(x), -1)
        label = self.classifier2(x)

        return logits, label


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device = torch.device('cuda')

# model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
# processor = Wav2Vec2Processor.from_pretrained(model_name)
# model = EmotionModel.from_pretrained(model_name)
# # model.classifier= myReg()
#
# def process_func(
#         x: np.ndarray,
#         sampling_rate: int,
#         embeddings: bool = False,
# ) -> np.ndarray:
#     r"""Predict emotions or extract embeddings from raw audio signal."""
#     # run through processor to normalize signal
#     # always returns a batch, so we just get the first entry
#     # then we put it on the device
#     y = processor(x, sampling_rate=sampling_rate)
#     y = y['input_values'][0]
#     # run through model
#     # with torch.no_grad():
#     y = torch.from_numpy(y).to(device)
#     ret = model(y)[0 if embeddings else 1]
#     ret = ret.to(device)
#     # convert to numpy
#     # y = y.detach().cpu().numpy()
#     return ret


# if dataset=='IEMOCAP1':
#     X_train,X_test,y_train,y_test = get_raw_transformer()
# elif dataset=='SAVEE':
#     X_train,X_test,y_train,y_test = get_raw_SAVEE_transformer()
print(len(X_train),len(X_test),len(y_train),len(y_test))
print(np.unique(y_train),np.unique(y_test))
print('loader built')

def Train_for_list(model, dl_train,dl_test,loss_function,loss_function2, metric_function, optimizer, epochs=150, batch_size=16,model_name='CNN_dd',df_path=folder+f'\\{dataset}_{clf}.csv'):
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
        # step = 1
        CM_train = torch.zeros((output_size,output_size)).int().to(device)
        CM_val = torch.zeros((output_size, output_size)).int().to(device)
        for step, (features,y_true,dim_true) in enumerate(dl_train,1):
            #print(seq.size(),labels.size())

            # optimizer.zero_grad()
            dim_pred, y_pred = model(features)

            # print(y_pred.size(),y_true.size())
            # print(len(y_pred),len(y_true))
            loss = w1* loss_function(y_pred, y_true) + w2* loss_function2(dim_pred,dim_true)
            loss.backward()
            if step % k == 0 or step == len(dl_train):
                optimizer.step()
                optimizer.zero_grad()
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
        for step, (features,labels,dim_true) in enumerate(dl_test,1):

            with torch.no_grad():
                dim_pred, y_pred = model(features)
                predictions=y_pred

                val_loss = w1* loss_function(predictions, labels) + w2* loss_function2(dim_pred,dim_true)
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
            model_save_path = folder + f'\\{dataset}_{clf}_{val_metric_sum / val_step:.3f}_loss={min_loss:.3f}.pth'.replace(
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

def Train_without_loader(model, X_train,X_test,y_train,y_test,loss_function,loss_function2, metric_function, optimizer, epochs=150, batch_size=16,model_name='CNN_dd',df_path=folder+f'\\{dataset}_{clf}.csv'):
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
        # step = 1
        CM_train = torch.zeros((output_size,output_size)).int().to(device)
        CM_val = torch.zeros((output_size, output_size)).int().to(device)
        for step, (features,y_true,dim_true) in enumerate(zip(X_train,y_train[:,None],dim_train),1):
            #print(seq.size(),labels.size())
            features = torch.from_numpy(features[None,:]).to(device)
            y_true = torch.from_numpy(y_true[None,:]).type(torch.LongTensor).to(device)[0]
            dim_true = torch.from_numpy(dim_true[None,:]).to(device)
            # optimizer.zero_grad()
            dim_pred, y_pred = model(features)
            # print(y_true.size(),y_pred.size())
            # print(y_pred.size(),y_true.size(),dim_pred.shape,dim_true.shape)
            # print(len(y_pred),len(y_true))
            loss = w1* loss_function(y_pred, y_true) + w2* loss_function2(dim_pred,dim_true)
            loss.backward()
            if step % k == 0 or step == len(X_train):
                optimizer.step()
                optimizer.zero_grad()
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
        # print('here',len(X_test),len(y_test))
        for val_step, (features,labels,dim_true) in enumerate(zip(X_test,y_test[:,None],dim_test),1):

            with torch.no_grad():
                features = torch.from_numpy(features[None,:]).to(device)
                labels = torch.from_numpy(labels[None,:]).type(torch.LongTensor).to(device)[0]
                dim_true = torch.from_numpy(dim_true[None,:]).to(device)
                dim_pred, y_pred = model(features)
                predictions = y_pred

                val_loss = w1* loss_function(predictions, labels) + w2* loss_function2(dim_pred,dim_true)
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

        if loss_sum / step < min_loss:
            min_loss = loss_sum / step
            model_save_path = folder + f'\\{dataset}_{clf}_{val_metric_sum / val_step:.3f}_loss={min_loss:.3f}.pth'.replace(
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

model,min_loss,max_acc,model_list = load_model(folder)
if model is None:
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    model = EmotionModel.from_pretrained(model_name)
    model = model.to(device)
# CNN = torch.load(r"E:\FYP excel\IEMOCAP1_encoder_CNN\IEMOCAP1_CNN_MFCC30_402_loss=1.816.pth").to(device)
if test_mode:
    model = load_best_model(folder).to(device)
model.classifier2[0] = nn.Dropout(p=dropout,inplace=False)
model.classifier2[3] = nn.Dropout(p=dropout,inplace=False)
print(f'model={model}')
metric_func = accuracy
metric_name = "accuracy"
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
loss_function2 = nn.MSELoss()
loss_function2 = loss_function2.to(device)
def loss_func(y_pred,y_true):
    one_hot = torch.zeros((len(y_true),output_size)).to(device)
    for a,b in zip(one_hot,y_true):
        a[b]=1
    return loss_function(y_pred,one_hot)


optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

Train_without_loader(model,X_train,X_test,y_train,y_test,batch_size=batch_size,loss_function=loss_function,loss_function2=loss_function2,metric_function=metric_func,optimizer=optimizer,epochs=10000000000)



