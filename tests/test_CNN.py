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
get_mfcc(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav')
# get_mfcc(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\sentences\\wav')
# get_mfcc(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\sentences\\wav')
# get_mfcc(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\sentences\\wav')
print(signals[0].shape)

flags,label_cat=[],[]
get_label_4cat(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\dialog\\EmoEvaluation')
# get_label_4cat(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\dialog\\EmoEvaluation')
# get_label_4cat(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\dialog\\EmoEvaluation')
# get_label_4cat(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\dialog\\EmoEvaluation')
# get_label_4cat(label_cat,flags,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\dialog\\EmoEvaluation')

flags = np.array(flags)
signals = np.array(signals)
idx = flags == 1
signals = signals[idx]
print(len(signals))

X, X_test, y, y_test = train_test_split(signals, label_cat, test_size=0.1,random_state=0)
X_train, X_val,y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=0)
y_train = torch.tensor(y_train).to(device)
# X_test = torch.tensor(X_test).type(torch.float32).to(device)
y_test = torch.tensor(y_test).to(device)
# X_val = torch.tensor(X_val).type(torch.float32).to(device)
y_val = torch.tensor(y_val).to(device)

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)

class CNN_Model(nn.Module):
    def __init__(self,input_size, hidden_layer=[1024,512,128],kernel_len=[100,80,60,40,20,10,5], channel=256, k=10, output_size=4):
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
        x = torch.nn.functional.softmax(x)
        return x

prob = torch.tensor([0.65163934, 0.4605621 , 0.45110701, 0.])

def train(model, X_train, y_train,loss_function, optimizer,  epochs=150, batch_size=32,path='D:\\pycharmProject\\FYP'):
    global prob
    for i in trange(epochs):
        for b in range(len(X_train)//batch_size):
            #print(seq.size(),labels.size())
            batch_X = X_train[b*batch_size:b*batch_size+batch_size]
            batch_X = torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True, padding_value=0).unsqueeze(1).to(device)
            labels = y_train[b*batch_size:b*batch_size+batch_size]
            ids = torch.argmax(labels, axis=1)
            probs = prob[ids]
            rands = torch.rand(len(batch_X))
            idx = rands > probs
            batch_X = batch_X[idx]
            # print(len(batch_X))
            labels = labels[idx]
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
        model.eval()
        cnt = np.zeros((4,))
        with torch.no_grad():
            error_num, num=0 , 0
            for b in tqdm(range(len(X_test) // batch_size - 1)):
                batch_X = X_test[b * batch_size:b * batch_size + batch_size]
                batch_X = torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True, padding_value=0).unsqueeze(1).to(
                    device)
                y_pred = model(batch_X)
                y_pred = y_pred.detach().cpu().numpy()
                label = y_test[b * batch_size:b * batch_size + batch_size]
                label = label.detach().cpu().numpy()
                idx1 = np.argmax(y_pred, axis=1)
                idx2 = np.argmax(label, axis=1)
                idx = idx1 != idx2
                lb , cnt1 = np.unique(idx1[idx], return_counts=True)
                for j in range(len(cnt1)):
                    cnt[lb[j]] += cnt1[j]
                # print(cnt)
                error_num += np.unique(idx, return_counts=True)[1][1]
                num += batch_size
        print(cnt)
        cnt = cnt / np.sum(cnt) - 0.2
        prob += cnt / 20
        model.train()
        print(f'new_prob={prob},acc={1-error_num/num}')
        torch.save(model, path)

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
        batch_X =  torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True, padding_value=0).unsqueeze(1).to(device)
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


print(f'signals[0].shape={signals[0].shape}')
if type(signals[0]) ==np.ndarray:
    input_size = signals[0].shape[-1]
else:
    input_size = signals[0].size()[-1]
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
mode ='train'
if mode  =='train':
    model = CNN_Model(input_size=input_size, hidden_layer=[512,256,128],kernel_len=[100,80,60,40,20,10,5], channel=8, k=20, output_size=4)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    train(model,X_train,y_train,loss_function,optimizer,epochs=150, path='D:\\pycharmProject\\FYP\\CNN3_ch1_cat4_mfcc.pth')
else:
    model = torch.load('D:\\pycharmProject\\FYP\\CNN3_ch1_cat4.pth')
loss, error_rate = eval(model,loss_function,X_test,y_test,batch_size=32)
save_path = 'D:\\pycharmProject\\FYP\\CNN_ch'+str(model.channel)+'_k'+str(model.k)+'_acc'+str(int(1-error_rate))+'.pth'
torch.save(model,save_path)
print(f'loss={loss},error rate={error_rate}')
