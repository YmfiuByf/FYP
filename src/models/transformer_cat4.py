# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.model_selection import train_test_split
# from transformers import Wav2Vec2Processor
# from transformers.models.wav2vec2.modeling_wav2vec2 import (
#     Wav2Vec2Model,
#     Wav2Vec2PreTrainedModel,
# )
# import os
# from tqdm import *
# import librosa
# from torch.optim import *
# from preprocessing import *
#
#
# class myReg(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.dense = nn.Linear(1024, 512)
#         self.dropout = nn.Dropout(0.3)
#         self.out_proj = nn.Linear(512, 4)
#         # self.softmax = nn.Softmax()
#
#     def forward(self, features, **kwargs):
#         x = features
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         # x = self.softmax(x)
#         x = torch.nn.functional.softmax(x,-1)
#
#         return x
#
# class RegressionHead(nn.Module):
#     r"""Classification head."""
#
#     def __init__(self, config):
#
#         super().__init__()
#
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dropout = nn.Dropout(config.final_dropout)
#         self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
#
#     def forward(self, features, **kwargs):
#
#         x = features
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#
#         return x
#
#
# class EmotionModel(Wav2Vec2PreTrainedModel):
#     r"""Speech emotion classifier."""
#
#     def __init__(self, config):
#
#         super().__init__(config)
#
#         self.config = config
#         self.wav2vec2 = Wav2Vec2Model(config)
#         self.classifier = RegressionHead(config)
#         self.init_weights()
#
#     def forward(
#             self,
#             input_values,
#     ):
#
#         outputs = self.wav2vec2(input_values)
#         hidden_states = outputs[0]
#         hidden_states = torch.mean(hidden_states, dim=1)
#         logits = self.classifier(hidden_states)
#
#         return hidden_states, logits
#
#
# if True:
#     k=1
#     # load model from hub
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     #device = torch.device('cuda')
#
#     # dummy signal
#
#
#
#     def process_func(
#         x: np.ndarray,
#         sampling_rate: int,
#         embeddings: bool = False,
#     ) -> np.ndarray:
#         r"""Predict emotions or extract embeddings from raw audio signal."""
#
#         # run through processor to normalize signal
#         # always returns a batch, so we just get the first entry
#         # then we put it on the device
#         y = processor(x, sampling_rate=sampling_rate)
#         y = y['input_values'][0]
#         # run through model
#         #with torch.no_grad():
#         y = torch.from_numpy(y).to(device)
#         ret = model(y)[0 if embeddings else 1]
#         ret = ret.to(device)
#         # convert to numpy
#         # y = y.detach().cpu().numpy()
#         return ret
#
#
#
#
#
#     print(torch.cuda.is_available())
#
#     #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#     signals=[]
#     get_original_signal(signals,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{k}\\sentences\\wav')
#     # get_original_signal(signals,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{k}\\sentences\\wav')
#     # get_original_signal(signals,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{k}\\sentences\\wav')
#     # get_original_signal(signals,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{k}\\sentences\\wav')
#     # get_original_signal(signals,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{k}\\sentences\\wav')
#     # print(len(signals))
#     # test = torch.nn.utils.rnn.pad_sequence(signals[0:16], batch_first=True)
#     label_cat,flags=[],[]
#     get_label_4cat(label_cat,flags,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{k}\\dialog\\EmoEvaluation')
#     # get_label_4cat(label_cat,flags,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{k}\\dialog\\EmoEvaluation')
#     # get_label_4cat(label_cat,flags,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{k}\\dialog\\EmoEvaluation')
#     # get_label_4cat(label_cat,flags,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{k}\\dialog\\EmoEvaluation')
#     # get_label_4cat(label_cat,flags,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{k}\\dialog\\EmoEvaluation')

#     flags = np.array(flags)
#     signals = np.array(signals)
#     idx = flags==1
#     signals = signals[idx]
#     print(len(signals))
#     X, X_test, y, y_test = train_test_split(signals, label_cat, test_size=0.1,random_state=0)
#     X_train, X_val,y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=0)
#
#     # if t >len(X_train):
#     #     return
#     # ed = min(t+40,len(X_train))
#     # X_train = X_train[t:ed]
#     # y_train = y_train[t:ed]
#
#     model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
#     processor = Wav2Vec2Processor.from_pretrained(model_name)
#     model = EmotionModel.from_pretrained(model_name)
#     model.classifier= myReg()
#     # model = torch.load('D:\\pycharmProject\\FYP\\transformer_cat4_freezeCNN_12345.pth')
#     # 冻结CNN
#     for p in model.wav2vec2.feature_extractor.conv_layers.parameters():
#         p.require_grad = False
#     for i in range(12):
#         for p in model.wav2vec2.encoder.layers[i].parameters():
#             p.require_grad = False
#     print(model)
#     model = model.to(device)
#
#
#
#     def train(model, X_train, y_train,loss_function, optimizer,  epochs=150, batch_size=2,path='D:\\pycharmProject\\FYP'):
#         model.train()
#         for i in trange(epochs):
#             for b in range(len(X_train)//batch_size -1 ):
#
#                 if batch_size == 1:
#                     x = X_train[b][None, :]
#                     label = torch.tensor(y_train[b][None, :]).type(torch.float32).to(device)
#                     y_pred = process_func(x, sampling_rate=16000)
#                     # print(y_pred,label)
#                 else:
#                     y_pred = []
#                     for j in range(b * batch_size, b * batch_size + batch_size):  # len(X_test)):
#                         y_pred.append(process_func(X_train[j][None, :], 16000)[0])
#                     y_pred = torch.stack(y_pred, 0)
#                     label = torch.tensor(y_test[b * batch_size: b * batch_size + batch_size]).type(torch.float32).to(device)
#                 # optimizer.zero_grad()
#                 single_loss = loss_function(y_pred, label)
#                 single_loss.backward()
#                 optimizer.step()
#                 del single_loss
#
#                 # x = X_train[b][None,:]
#                 # label = torch.tensor( y_train[b][None,:]).type(torch.float32).to(device)
#                 # x = processor(x, sampling_rate=16000)
#                 # x = x['input_values'][0]
#                 # # run through model
#                 # # with torch.no_grad():
#                 # x = torch.from_numpy(x).to(device)
#                 # y_pred = model(x)[1]
#                 # y_pred = y_pred.to(device)
#                 # # y_pred = process_func(x, sampling_rate=16000)
#                 # # print(y_pred,label)
#                 # single_loss = loss_function(y_pred,label)
#                 # single_loss.backward()
#                 # optimizer.step()
#                 # # print(label,y_pred)
#                 # label.cpu()
#                 # y_pred.cpu()
#                 # x.cpu()
#                 # del label,y_pred,x
#
#             if i%25 == 5:
#                 print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
#             # print(f'epoch={i}')
#             torch.save(model, path)
#             # scheduler.step()
#         torch.save(model, path)
#         return
#
#     def eval(model,dl_val,loss_function):
#         model.eval()
#         error_num, num, total_loss = 0, 0, 0
#
#         for step, data in enumerate(dl_val):
#             x, y = data
#             y_pred = model(x)
#             loss = loss_function(y_pred, y)
#             errors = torch.argmax(y_pred.cpu(), dim=1) == torch.argmax(y.cpu(), dim=1)
#             error_num += np.unique(errors, return_counts=True)[1][0]
#             single_loss = loss_function(y_pred, y)  # 损失函数
#             total_loss += single_loss
#             num += len(x)
#         model.train()
#         return total_loss/num, error_num/num
#
#
#     class CCCLoss(nn.Module):  # [batch_size , 3]
#         def __init__(self):
#             super().__init__()
#
#         def forward(self,y_pred, y_gt):
#             x = y_pred.permute(1,0)
#             y = y_gt.permute(1,0)
#             z0 = [x[0][None,:], y[0][None,:]]
#             z1 = [x[1][None,:], y[1][None,:]]
#             z2 = [x[2][None,:], y[2][None,:]]
#             z0 = torch.cat(z0,dim=0)
#             z1 = torch.cat(z1,dim=0)
#             z2 = torch.cat(z2,dim=0)
#             m0 = torch.mean(z0, dim=1)
#             m1 = torch.mean(z1, dim=1)
#             m2 = torch.mean(z2, dim=1)
#             s0 = torch.cov(z0)
#             s1 = torch.cov(z1)
#             s2 = torch.cov(z2)
#             def ccc(s,m):
#                 return 2*s[0,1]**2/( (m[0]-m[1])**2 + s[0,0]**2 + s[1,1]**2 )
#             return ( ccc(s0,m0) + ccc(s1,m1) + ccc(s2,m2) ) / 3
#
#
#     # loss_function = CCCLoss()
#     loss_function = nn.CrossEntropyLoss()
#     # loss_function = loss_function.to(device)
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6)
#     # scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.8)
#     # print(model.wav2vec2.encoder)
#     # for layer in model.wav2vec2.encoder.layers:
#     #     print(layer)
#     train(model, X_train, y_train, loss_function, optimizer, epochs=100, batch_size=1, path='D:\\pycharmProject\\FYP\\transformer_cat4_WTF1.pth')
#
#     # loss, error_rate = eval(model,dl_val,loss_function)
#     # print(f'loss={loss},error rate={error_rate}')
#
# # if __name__=='__main__':
# #
# #     for epoch in range(1):
# #         for k in [5]:
# #             for t in [120]:
# #                 print(f'epoch={epoch},k={k}')
# #                 main(k,t)
#
# ##########################################

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

from preprocessing import *

def main(k,load_path,save_path,prior_prob):
    # load model from hub
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #device = torch.device('cuda')

    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    # model = EmotionModel.from_pretrained(model_name)
    # model.classifier= myReg()
    model = torch.load(load_path)
    model.to(device)
    # 冻结CNN
    for p in model.wav2vec2.feature_extractor.conv_layers.parameters():
        p.require_grad = False
    print(model)

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
    if k==0:
        get_original_signal(signals, f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav')
        get_original_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\sentences\\wav')
        get_original_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\sentences\\wav')
        get_original_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\sentences\\wav')
        get_original_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\sentences\\wav')
    else:
        get_original_signal(signals,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{k}\\sentences\\wav')
    print(len(signals))
    # test = torch.nn.utils.rnn.pad_sequence(signals[0:16], batch_first=True)
    label_cat,flags=[],[]
    if k==0:
        get_label_4cat(label_cat, flags, f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\dialog\\EmoEvaluation')
        get_label_4cat(label_cat,flags,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\dialog\\EmoEvaluation')
        get_label_4cat(label_cat,flags,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\dialog\\EmoEvaluation')
        get_label_4cat(label_cat,flags,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\dialog\\EmoEvaluation')
        get_label_4cat(label_cat,flags,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\dialog\\EmoEvaluation')
    else:
        get_label_4cat(label_cat,flags,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{k}\\dialog\\EmoEvaluation')

    flags = np.array(flags)
    signals = np.array(signals)
    # label_cat = np.array(label_cat)
    idx = flags==1
    signals = signals[idx]
    # label_cat = label_cat[idx]
    signals = signals.tolist()


    X_train, X_test, y_train, y_test = train_test_split(signals, label_cat, test_size=0.2,random_state=0)
    # X_train, X_val,y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=0)




    def train(model, X_train, y_train,loss_function, optimizer,  epochs=150, batch_size=2,path='D:\\pycharmProject\\FYP'):
        model.train()
        # prob = [0.6484375, 0.41048035, 0.30412371, 0.]
        # prob = [0.43958134, 0.50699608, 0.78592307, 0.11054107]
        prob = prior_prob
        for i in trange(epochs):
            model.train()
            for n in range(len(X_train)//batch_size -1 ):
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
                label = torch.tensor(y_train[n][None, :]).type(torch.float32).to(device)
                idx = torch.argmax(label)
                # print(idx,label)
                if np.random.rand() < prob[idx]:
                    continue
                x = X_train[n][None,:]
                y_pred = process_func(x, sampling_rate=16000)
                # print(y_pred,label)
                single_loss = loss_function(y_pred,label)
                optimizer.zero_grad()
                single_loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                error = 0
                num = len(X_test)
                cnt = np.zeros((4,))
                for j in range(num):
                    x = X_test[j][None, :]
                    # label = torch.tensor(y_test[i][None, :]).type(torch.float32).to(device)
                    label = np.array(y_test[j][None, :])
                    y_pred = process_func(x, sampling_rate=16000)
                    # loss = loss_function(y_pred, label)
                    # total_loss+=loss
                    y_pred = y_pred.detach().cpu().numpy()
                    # label  = label.detach().cpu().numpy()
                    # print(y_pred, label)
                    if np.argmax(y_pred) != np.argmax(label):
                        error += 1.
                        cnt[np.argmax(y_pred)] += 1
                    else:
                        # print(f'right {len(x[0])}')
                        pass
                print(f'epoch={i},acc={1 - error / num},count={cnt}')
            cnt = cnt/np.sum(cnt) - 0.2
            prob += cnt/20
            model.train()
            print(f'new_prob={prob}')
            torch.save(model,path)
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
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6)
    train(model, X_train, y_train, loss_function, optimizer, epochs=150, batch_size=1, path=save_path)

# loss, error_rate = eval(model,dl_val,loss_function)
# print(f'loss={loss},error rate={error_rate}')
if __name__ == '__main__':
    overall = False
    if overall==True:
        load_path = r"D:\pycharmProject\FYP\transformer_cat4_t3.pth"
        save_path =r"D:\pycharmProject\FYP\transformer_cat4_overall.pth"
        main(0 ,load_path, save_path,[0.65163934, 0.4605621 , 0.45110701, 0.])
    else:
        k=5
        prior_prob = [[0.6484375 , 0.41048035, 0.30412371, 0.],[0.67679558, 0.1459854 , 0.40609137, 0.],[0.578125  , 0.4375    , 0.55737705, 0.],[0.74806202, 0.80122324, 0.54545455, 0.],[0.62760417, 0.15882353, 0.41632653, 0.]]
        load_path = f"D:\\pycharmProject\\FYP\\transformer_cat4_t{k-1}.pth"
        save_path = f"D:\\pycharmProject\\FYP\\transformer_cat4_t{k}.pth"
        main(k,load_path, save_path,prior_prob[k-1])
