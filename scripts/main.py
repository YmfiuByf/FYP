from preprocessing import *
from DBN import *
from wav2vec import *
from RBM import *

import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda
import matplotlib.pyplot as plt
import numpy as np
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence


#print(model)
#print(processor)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
signals=[]
get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav')
print(signals[0].size()[1],signals[1].size())
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\sentences\\wav')
print(len(signals))


label_dim,label_cat=[],[]
get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\dialog\\EmoEvaluation')
# get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\dialog\\EmoEvaluation')
# get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\dialog\\EmoEvaluation')
# get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\dialog\\EmoEvaluation')
# get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\dialog\\EmoEvaluation')

# print( torch.cuda.is_available() )
# device = 'gpu'
# model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
# processor = Wav2Vec2Processor.from_pretrained(model_name)
# model = EmotionModel.from_pretrained(model_name)

X, X_test, y, y_test = train_test_split(signals, label_cat, test_size=0.1,random_state=0)
X_train, X_val,y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=0)
y_train = torch.tensor(y_train).to(device)
y_val = torch.tensor(y_val).to(device)
y_test = torch.tensor(y_test).to(device)
# X_train_len = [len(sq) for sq in X_train]
# X_train = pad_sequence(X_train).transpose(1,0)
# X_val_len = [len(sq) for sq in X_val]
# X_val = pad_sequence(X_val).transpose(1,0)
# print(X_train.size())
# print(X_train[0,-1], X_train[100,-1])
# # x_train y_train 和 x_test y_test都是经过预处理的DataFrame数据
# dl_train = DataLoader( TensorDataset(X_train.clone().detach().requires_grad_(True), torch.tensor(np.array(y_train) ) ) ,  shuffle=True, batch_size=16 )
# dl_valid = DataLoader( TensorDataset(X_val.clone().detach().requires_grad_(True), torch.tensor(np.array(y_val) ) ), shuffle=True, batch_size=16  )



class LSTM(nn.Module):
    def __init__(self, input_size=13, hidden_layer_size=100, output_size=9, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,batch_size,self.hidden_layer_size),
                            torch.zeros(1,batch_size,self.hidden_layer_size))

    def forward(self, input_seq):
        #lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        lstm_out, self.hidden_cell = self.lstm(input_seq,self.hidden_cell)
        #predictions = self.linear(lstm_out.view(len(input_seq), -1))
        pred = self.linear( self.hidden_cell[0][0].squeeze() )
        return pred

model = LSTM(input_size=signals[0].size()[1], hidden_layer_size=50, output_size=9, batch_size=32)
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)

def train(model, X_train, y_train,loss_function, optimizer,  epochs=150, batch_size=32,path='D:\\pycharmProject\\FYP'):
    for i in range(epochs):
        for b in range(len(X_train)//batch_size):
            #print(seq.size(),labels.size())
            batch_X = X_train[b*batch_size:b*batch_size+batch_size]
            labels = y_train[b*batch_size:b*batch_size+batch_size]
            batch_X = pack_sequence(batch_X, enforce_sorted=False)
            # print(batch_X[0],batch_X[1],len(batch_X))
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, batch_size, model.hidden_layer_size).to(device),
                            torch.zeros(1, batch_size, model.hidden_layer_size).to(device))

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
        batch_X = pack_sequence(batch_X, enforce_sorted=False)
        # print(batch_X[0],batch_X[1],len(batch_X))
        model.hidden_cell = (torch.zeros(1, batch_size, model.hidden_layer_size).to(device),
                             torch.zeros(1, batch_size, model.hidden_layer_size).to(device))

        y_pred = model(batch_X)
        errors = torch.argmax(y_pred.cpu(),dim=1)==torch.argmax(labels.cpu(),dim=1)
        print(errors)
        error_num +=np.unique(errors,return_counts=True)[1][0]
        single_loss = loss_function(y_pred, labels)  # 损失函数
        total_loss += single_loss
        num+= batch_size

        if b%10==0:
            print(f'single loss ={single_loss}')
    return total_loss/num, error_num/num


train(model, X_train, y_train, loss_function, optimizer, epochs=150, batch_size=32, path='D:\\pycharmProject\\FYP\\LSTM4_onehot.pth')


b, batch_size = 0, 32
batch_X = X_test[b * batch_size:b * batch_size + batch_size]
batch_X = pack_sequence(batch_X, enforce_sorted=False)
# print(batch_X[0],batch_X[1],len(batch_X))
model.hidden_cell = (torch.zeros(1, batch_size, model.hidden_layer_size).to(device),
                     torch.zeros(1, batch_size, model.hidden_layer_size).to(device))
y_pred = model(batch_X)
print(torch.nn.functional.softmax(y_pred[0],dim=0),y_test[0])


total_loss, error_rate = eval(model,loss_function=loss_function ,X_val=X_val,y_val=y_val,batch_size=32)
print(f'total_loss={total_loss},error rate={error_rate}')


