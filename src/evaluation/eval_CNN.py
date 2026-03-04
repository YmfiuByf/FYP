import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
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
get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\sentences\\wav')
print(signals[0].shape)

label_cat, flags = [], []
get_label_4cat(label_cat, flags, f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\dialog\\EmoEvaluation')
# get_label_4cat(label_cat, flags, f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\dialog\\EmoEvaluation')
# get_label_4cat(label_cat, flags, f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\dialog\\EmoEvaluation')
# get_label_4cat(label_cat, flags, f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\dialog\\EmoEvaluation')
# get_label_4cat(label_cat, flags, f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\dialog\\EmoEvaluation')

flags = np.array(flags)
signals = np.array(signals)
idx = flags == 1
signals = signals[idx]
signals = signals.tolist()
print(len(signals))


X, X_test, y, y_test = train_test_split(signals, label_cat, test_size=0.1,random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=0)
# y_train = torch.tensor(np.array(y_train)).to(device)
# # X_test = torch.tensor(X_test).type(torch.float32).to(device)
# y_test = torch.tensor(y_test).to(device)
# # X_val = torch.tensor(X_val).type(torch.float32).to(device)
# y_val = torch.tensor(y_val).to(device)

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
            batch_X = torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True, padding_value=0).unsqueeze(1).to(device)
            optimizer.zero_grad()
            y_pred = model(batch_X)

            single_loss = loss_function(y_pred, labels)  #损失函数
            single_loss.backward() #前向
            optimizer.step()

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    torch.save(model, path)
    return

def eval(model, loss_function=None, X_train=None, y_train=None, batch_size=32):
    model.eval()
    error_num = 0
    total_loss = 0
    num=0
    if True:
        for b in tqdm(range(len(X_train)//batch_size -1 )):
            batch_X = X_train[b * batch_size:b * batch_size + batch_size]
            batch_X = torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True, padding_value=0).unsqueeze(1).to(
                device)
            y_pred = model(batch_X)
            y_pred = y_pred.detach().cpu().numpy()
            label = np.array(y_train[b*batch_size:b*batch_size+batch_size])
            idx1 = np.argmax(y_pred,axis=1)
            idx2 = np.argmax(label,axis=1)
            idx = idx1!=idx2
            print(np.unique(idx1[idx],return_counts=True),np.unique(idx2[idx],return_counts=True))
            error_num += np.unique(idx,return_counts=True)[1][1]
            num+= batch_size
    return total_loss/num, error_num/num


print(f'signals[0].shape={signals[0].shape}')
model = torch.load('D:\\pycharmProject\\FYP\\CNN3_ch1_cat4_3.pth')
mode = model.to(device)
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
loss, error_rate = eval(model,loss_function,X_val,y_val,batch_size=32)
print(error_rate)
save_path = 'D:\\pycharmProject\\FYP\\CNN_ch'+str(model.channel)+'_k'+str(model.k)+'_acc'+str(int(100*(1-error_rate)))+'.pth'
torch.save(model,save_path)
print(f'loss={loss},acc={1-error_rate}')
