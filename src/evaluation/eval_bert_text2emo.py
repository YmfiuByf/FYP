import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
import librosa
from transformers import BertTokenizer, BertModel
from preprocessing import *
from torch.optim import *
from tqdm.notebook import tqdm


class myReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 384)
        self.dropout = nn.Dropout(0.3)
        self.out_proj = nn.Linear(384, 4)
        self.softmax = nn.Softmax()
        self.activation =  nn.Tanh()

    def forward(self, features, **kwargs):
        x = features
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = self.softmax(x)

        return x


# load model from hub
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device = torch.device('cuda')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
# model.pooler.activation= myReg()
model = torch.load(r"D:\pycharmProject\FYP\transformer_cat4_text2emo_12345.pth")
model.to(device)
# 冻结CNN
for p in model.encoder.parameters():
    p.require_grad = False



print(torch.cuda.is_available())

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
signals=[]
get_text(signals,r"D:\pycharmProject\FYP\IEMOCAP语料库\Session1\dialog\transcriptions")
get_text(signals,r"D:\pycharmProject\FYP\IEMOCAP语料库\Session2\dialog\transcriptions")
get_text(signals,r"D:\pycharmProject\FYP\IEMOCAP语料库\Session3\dialog\transcriptions")
get_text(signals,r"D:\pycharmProject\FYP\IEMOCAP语料库\Session4\dialog\transcriptions")
get_text(signals,r"D:\pycharmProject\FYP\IEMOCAP语料库\Session5\dialog\transcriptions")
print(len(signals))
# test = torch.nn.utils.rnn.pad_sequence(signals[0:16], batch_first=True)
label_dim,label_cat,flags=[],[],[]
label_cat,flags=[],[]
get_label_4cat(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\dialog\\EmoEvaluation')
get_label_4cat(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\dialog\\EmoEvaluation')
get_label_4cat(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\dialog\\EmoEvaluation')
get_label_4cat(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\dialog\\EmoEvaluation')
get_label_4cat(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\dialog\\EmoEvaluation')
# test_ret =  process_func(test,16000)

flags = np.array(flags)
signals = np.array(signals)
idx = flags==1
signals = signals[idx]
print(len(signals))
# signals = signals[0:100]
# label_dim = label_dim[:100]

X, X_test, y, y_test = train_test_split(signals, label_cat, test_size=0.1,random_state=0)
X_train, X_val,y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=0)

def train(model, X_train, y_train,loss_function, optimizer,scheduler,  epochs=150, batch_size=2,path='D:\\pycharmProject\\FYP'):
    model.train()
    for i in tqdm(range(epochs)):
        for b in tqdm(range(len(X_train)//batch_size -1 )):

            # batch_X = torch.tensor( X_train[b*batch_size:b*batch_size+batch_size] ).to(device)
            # labels = y_train[b*batch_size:b*batch_size+batch_size]
            # labels = torch.tensor(labels).type(torch.float32).to(device)
            # batch_X = torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True)
            # # print(batch_X[0],batch_X[1],len(batch_X))
            # optimizer.zero_grad()
            # y_pred = process_func(batch_X, 16000)
            #
            # single_loss = loss_function(y_pred, labels)  #损失函数
            # single_loss.backward() #前向
            #
            # optimizer.step()
            texts = []
            text = X_train[b*batch_size:b*batch_size+batch_size]
            # print(text)
            for t in text:
                encoded_input = tokenizer(t, return_tensors='pt')['input_ids'][0]
                texts.append(encoded_input)
            texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0).to(device)
            y_pred = model(texts)[1]
            label = torch.tensor( y_train[b*batch_size:b*batch_size+batch_size]).type(torch.float32).to(device)
            # print(y_pred.size(),label.size())
            single_loss = loss_function(y_pred,label)
            single_loss.backward()
            optimizer.step()
        print(y_pred,label)
        print(single_loss)
        torch.save(model, path)
        scheduler.step()
        if i%25 == 5:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
            torch.save(model, path)
    torch.save(model, path)
    return

def eval(model,X_train,y_train,loss_function=None,batch_size=16):
    model.eval()
    error_num = 0
    total_loss = 0
    num=0
    if True:
        for b in tqdm(range(len(X_train)//batch_size -1 )):
            texts = []
            text = X_train[b*batch_size:b*batch_size+batch_size]
            # print(text)
            for t in text:
                encoded_input = tokenizer(t, return_tensors='pt')['input_ids'][0]
                texts.append(encoded_input)
            texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0).to(device)
            y_pred = model(texts)[1]
            y_pred = y_pred.detach().cpu().numpy()
            label = np.array(y_train[b*batch_size:b*batch_size+batch_size])
            idx1 = np.argmax(y_pred,axis=1)
            idx2 = np.argmax(label,axis=1)
            idx = idx1==idx2
            error_num += np.unique(idx,return_counts=True)[1][0]
            # label =  torch.tensor(label).type(torch.float32).to(device)
            # # print(y_pred.size(),label.size())
            # single_loss = loss_function(y_pred,label)
            num+= batch_size

    model.train()
    return total_loss/num, 1-error_num/num

# def get_label(y):
#     m = torch.argmax(y,dim=1)
#     ret = torch.zeros_like()
# loss_function = CCCLoss()
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
# _, err = eval(model,X_val,y_val,loss_function)
# print(len(X_test),f'acc={1-err}')
t = ['look at your grade, what are you doing']
encoded_input = tokenizer(t, return_tensors='pt')['input_ids'][0][None,:].to(device)
y_pred = model(encoded_input)[1]
print(y_pred)
ave_loss,acc = eval(model,X_train,y_train)
print(ave_loss,acc)
# loss, error_rate = eval(model,dl_val,loss_function)
# print(f'loss={loss},error rate={error_rate}')
