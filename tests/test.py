import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import LSTM
from preprocessing import *
from sklearn.model_selection import train_test_split
from teager import Teager
from torch import nn

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

model = torch.load('D:\\pycharmProject\\FYP\\LSTM4_onehot.pth')

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
signals=[]
get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav')
get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\sentences\\wav')
get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\sentences\\wav')
get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\sentences\\wav')
get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\sentences\\wav')
print(len(signals))


label_dim,label_cat=[],[]
get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\dialog\\EmoEvaluation')
get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\dialog\\EmoEvaluation')
get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\dialog\\EmoEvaluation')
get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\dialog\\EmoEvaluation')
get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\dialog\\EmoEvaluation')
# print(np.argmax(label_cat[0]), signals[1],Teager(signals[0],'horizontal',1) )#, Teager(signals[1],'horizontal',1))

X, X_test, y, y_test = train_test_split(signals, label_cat, test_size=0.1,random_state=0)
X_train, X_val,y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=0)
y_train = torch.tensor(y_train).to(device)
y_test = torch.tensor(y_test).to(device)
y_val = torch.tensor(y_val).to(device)

b, batch_size = 0, 32
batch_X = X_test[b * batch_size:b * batch_size + batch_size]
batch_X = pack_sequence(batch_X, enforce_sorted=False)
# print(batch_X[0],batch_X[1],len(batch_X))
model.hidden_cell = (torch.zeros(1, batch_size, model.hidden_layer_size).to(device),
                     torch.zeros(1, batch_size, model.hidden_layer_size).to(device))
y_pred = model(batch_X)
print(torch.nn.functional.softmax(y_pred[0],dim=0),y_test[0])

loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
loss, error_rate = eval(model,loss_function=loss_function ,X_val=X_val,y_val=y_val,batch_size=32)
print(loss, error_rate)
