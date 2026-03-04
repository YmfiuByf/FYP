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
import numpy as np
import torch
from preprocessing import *
from torch.optim import *
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

class myReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.3)
        self.out_proj = nn.Linear(1024, 4)
        self.softmax = nn.Softmax()

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = self.softmax(x)

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

    def forward(
            self,
            input_values,
    ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


def main(s):




    # load model from hub
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda')

    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = torch.load(r"D:\pycharmProject\FYP\transformer_cat4_t4.pth")
    model.to(device)
    # 冻结CNN
    for p in model.wav2vec2.feature_extractor.conv_layers.parameters():
        p.require_grad = False
    # print(model)

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






    print(torch.cuda.is_available())

    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    signals=[]
    get_original_signal(signals,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{s}\\sentences\\wav')
    # get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\sentences\\wav')
    # get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\sentences\\wav')
    # get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\sentences\\wav')
    # get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\sentences\\wav')
    print(len(signals))
    # test = torch.nn.utils.rnn.pad_sequence(signals[0:16], batch_first=True)
    label_cat,flags=[],[]
    get_label_4cat(label_cat,flags,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{s}\\dialog\\EmoEvaluation')
    # get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\dialog\\EmoEvaluation')
    # get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\dialog\\EmoEvaluation')
    # get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\dialog\\EmoEvaluation')
    # get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\dialog\\EmoEvaluation')
    #
    # test_ret =  process_func(test,16000)

    flags = np.array(flags)
    signals = np.array(signals)
    # label_cat = np.array(label_cat)
    idx = flags == 1
    signals = signals[idx]
    # label_cat = label_cat[idx]
    signals = signals.tolist()

    # X, X_test, y, y_test = train_test_split(signals, label_cat, test_size=0.1,random_state=0)
    # X_train, X_val,y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=0)
    X_test = signals
    y_test = label_cat





    def train(model, X_train, y_train,loss_function, optimizer,scheduler,  epochs=150, batch_size=2,path='D:\\pycharmProject\\FYP'):
        model.train()
        for i in trange(epochs):
            for b in range(len(X_train)//batch_size -1 ):

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

                x = X_train[b][None,:]
                label = torch.tensor( y_train[b][None,:]).type(torch.float32).to(device)
                y_pred = process_func(x, sampling_rate=16000)
                print(y_pred,label)
                single_loss = loss_function(y_pred,label)
                single_loss.backward()
                optimizer.step()

            if i%25 == 5:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
            print(f'epoch={i}')
            torch.save(model, path)
            scheduler.step()
        torch.save(model, path)
        return

    def eval(model,dl_val,loss_function):
        model.eval()
        error_num, num, total_loss = 0, 0, 0

        for step, data in enumerate(dl_val):
            x, y = data
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            errors = torch.argmax(y_pred.cpu(), dim=1) == torch.argmax(y.cpu(), dim=1)
            error_num += np.unique(errors, return_counts=True)[1][0]
            single_loss = loss_function(y_pred, y)  # 损失函数
            total_loss += single_loss
            num += len(x)
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

    # loss_function = CCCLoss()
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)
    error = 0.
    num = len(X_test)
    # num=100
    total_loss = 0.
    y_preds = []
    y_gts = []
    for i in range(num):
        x = X_test[i][None, :]
        label = np.array(y_test[i][None, :])
        y_pred = process_func(x, sampling_rate=16000)
        y_pred = y_pred.detach().cpu().numpy()
        y_preds.append(np.argmax(y_pred))
        y_gts.append(np.argmax(label))
        if np.argmax(y_pred) != np.argmax(label):
            error+=1.
            print(f'y_pred={np.argmax(y_pred)},y_gt={np.argmax(label)}')
        else:
            # print(f'right {len(x[0])}')
            pass
    y_preds = np.array(y_preds)
    y_gts = np.array(y_gts)
    accuracy = 1-error/num
    cm = confusion_matrix(y_gts, y_preds, labels=np.array([0,1,2,3]))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Neural', 'Angry', 'Sad', 'Happy'])
    disp.plot()
    plt.title(f'Transformer,accuracy={accuracy}')
    plt.savefig(fr"D:\pycharmProject\FYP\mfccF0_CM_Transformer.png")
    print(f's={s},acc={accuracy}')

    # loss, error_rate = eval(model,dl_val,loss_function)
    # print(f'loss={loss},error rate={error_rate}')

if __name__=='__main__':
    for s in [5]:
        main(s)
