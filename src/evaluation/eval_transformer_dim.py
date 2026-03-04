import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
from tqdm import *
import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import os
from tqdm import *
import librosa




# load model from hub
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')



from preprocessing import *
print(torch.cuda.is_available())

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
signals=[]
get_original_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\sentences\\wav')
print(signals[0].shape)

label_dim,label_cat=[],[]
get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\dialog\\EmoEvaluation')
# get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\dialog\\EmoEvaluation')
# get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\dialog\\EmoEvaluation')
# get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\dialog\\EmoEvaluation')
# get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\dialog\\EmoEvaluation')


X, X_test, y, y_test = train_test_split(signals, label_dim, test_size=0.1,random_state=0)
X_train, X_val,y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=0)
# y_train = torch.tensor(y_train).to(device)
# # X_test = torch.tensor(X_test).type(torch.float32).to(device)
# y_test = torch.tensor(y_test).to(device)
# # X_val = torch.tensor(X_val).type(torch.float32).to(device)
# y_val = torch.tensor(y_val).to(device)

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

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits



# load model from hub
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device = torch.device('cuda')

model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)

# dummy signal



def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    # run through model
    #with torch.no_grad():
    y = torch.from_numpy(y).to(device)
    ret = model(y)[0 if embeddings else 1]
    ret = ret.to(device)
    # convert to numpy
    # y = y.detach().cpu().numpy()
    return ret
# y_train = torch.tensor(y_train).to(device)
# y_val = torch.tensor(y_val).to(device)
# y_test = torch.tensor(y_test).to(device)

from torch.utils.data import DataLoader, TensorDataset


# dl_train = DataLoader(TensorDataset(torch.tensor(X_train).type(torch.float32).to(device),torch.tensor(y_train).type(torch.float32).to(device)),batch_size=32)
# dl_test = DataLoader(TensorDataset(torch.tensor(X_test).type(torch.float32).to(device),torch.tensor(y_test).type(torch.float32).to(device)),batch_size=32)
# dl_val = DataLoader(TensorDataset(torch.tensor(X_val).type(torch.float32).to(device),torch.tensor(y_val).type(torch.float32).to(device)),batch_size=32)

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)

class CNN_Model(nn.Module):
    def __init__(self,input_size, hidden_layer=[1024,512,128],kernel_len=[100,80,60,40,20,10,5], channel=16, k=10, output_size=9):
        super().__init__()
        self.k = k
        self.channel = channel
        self.kernel_len = kernel_len
        self.activation = nn.ReLU()
        self.dense_layers = []
        self.num_hidden = len(hidden_layer)
        self.convs = []
        self.num_kernel = len(kernel_len)
        self.num_layer = len(hidden_layer)
        for i in range(self.num_layer):
            if i==0:
                input = self.num_kernel* self.k*self.channel
            else:
                input = hidden_layer[i-1]
            self.dense_layers.append(nn.Linear(input, hidden_layer[i]).to(device))
        for i in range(self.num_kernel):
            self.convs.append(nn.Conv2d(1,channel, kernel_size=(kernel_len[i],input_size) ).to(device) )
        self.output_layer = nn.Linear(hidden_layer[-1],output_size)

    def forward(self,input):
        x = torch.zeros((input.size()[0],0)).to(device)
        for conv in self.convs:
            tmp = conv(input)
            tmp = kmax_pooling(tmp,dim=-2, k=self.k)
            tmp = tmp.view(tmp.size()[0],-1)
            x = torch.cat((x,tmp), dim=1)
        for layer in self.dense_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.output_layer(x)
        return x

def train(model, X_train, y_train,loss_function, optimizer,  epochs=150, batch_size=32,path='D:\\pycharmProject\\FYP'):
    for i in trange(epochs):
        for b in range(len(X_train)//batch_size):
            #print(seq.size(),labels.size())
            batch_X = X_train[b*batch_size:b*batch_size+batch_size]
            labels = y_train[b*batch_size:b*batch_size+batch_size]
            batch_X = torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True, padding_value=0).unsqueeze(1)
            # print(f'batch_X size={batch_X.size()}')
            # print(batch_X[0],batch_X[1],len(batch_X))
            optimizer.zero_grad()


            y_pred = model(batch_X)

            single_loss = loss_function(y_pred, labels)  #损失函数
            single_loss.backward() #前向
            optimizer.step()

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    torch.save(model, path)
    return

def eval(model, loss_function, X_val, y_val, batch_size):
    model.eval()
    num=0
    total_loss=0
    error_num = 0
    for b in range(len(X_val) // batch_size):
        # print(seq.size(),labels.size())
        batch_X = X_val[b * batch_size:b * batch_size + batch_size]
        labels = y_val[b * batch_size:b * batch_size + batch_size]
        batch_X =  torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True, padding_value=0).unsqueeze(1)
        # print(batch_X[0],batch_X[1],len(batch_X))

        y_pred = model(batch_X)
        errors = torch.argmax(y_pred.cpu(),dim=1)==torch.argmax(labels.cpu(),dim=1)
        print(errors)
        error_num +=np.unique(errors,return_counts=True)[1][0]
        single_loss = loss_function(y_pred, labels)  # 损失函数
        total_loss += single_loss
        num+= batch_size

        if b%10==0:
            print(f'single loss ={single_loss}')
    model.train()
    return total_loss/num, error_num/num

class CCCLoss(nn.Module):  # [batch_size , 3]
    def __init__(self):
        super().__init__()

    def forward(self,y_pred, y_gt):
        x = y_pred.permute(1,0)
        y = y_gt.permute(1,0)
        z0 = [x[0][None,:], y[0][None,:]]
        z1 = [x[1][None,:], y[1][None,:]]
        z2 = [x[2][None,:], y[2][None,:]]
        z0 = torch.cat(z0,dim=0)
        z1 = torch.cat(z1,dim=0)
        z2 = torch.cat(z2,dim=0)
        m0 = torch.mean(z0, dim=1)
        m1 = torch.mean(z1, dim=1)
        m2 = torch.mean(z2, dim=1)
        s0 = torch.cov(z0)
        s1 = torch.cov(z1)
        s2 = torch.cov(z2)
        def ccc(s,m):
            return 2*s[0,1]**2/( (m[0]-m[1])**2 + s[0,0]**2 + s[1,1]**2 )
        return ( ccc(s0,m0) + ccc(s1,m1) + ccc(s2,m2) ) / 3


print(f'signals[0].shape={signals[0].shape}')
model = torch.load(r"D:\pycharmProject\FYP\transformer_dim_freezeCNN_CCCLoss.pth")
mode = model.to(device)
CCCLoss = CCCLoss()
y_pred = []
for i in range(10):#len(X_test)):
    y_pred.append(process_func(X_test[i][None,:],16000)[0])
y_pred = torch.stack(y_pred,0)
print(y_pred.size())
loss = CCCLoss(y_pred,torch.tensor( y_test[:10]).to(device) )
print(loss)
print(f'y_pred = {y_pred},y={y_test[:10]}')
# loss_function = nn.CrossEntropyLoss()
# loss_function = loss_function.to(device)
# loss, error_rate = eval(model,loss_function,X_val,y_val,batch_size=32)
# save_path = 'D:\\pycharmProject\\FYP\\CNN_ch'+str(model.channel)+'_k'+str(model.k)+'_acc'+str(int(100*(1-error_rate)))+'.pth'
# torch.save(model,save_path)
# print(f'loss={loss},error rate={error_rate}')
