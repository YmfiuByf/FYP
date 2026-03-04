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
model = torch.load('D:\\pycharmProject\\FYP\\transformer_dim_MSE.pth')
model.to(device)

# dummy signal
sampling_rate = 16000
signal = np.zeros((1, sampling_rate), dtype=np.float32)


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
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()


    return y


process_func(signal, sampling_rate)
#  Arousal    dominance valence
# [[0.5460759 0.6062269 0.4043165]]

print (process_func(signal, sampling_rate, embeddings=True) )
# Pooled hidden states of last transformer layer
# [[-0.00752167  0.0065819  -0.00746339 ...  0.00663631  0.00848747
#   0.00599209]]

def get_signal_transformer(signals,fp, fs=16000):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav_file_path = fp
    orig_wav_files = os.listdir(wav_file_path)
    for orig_wav_file in tqdm(orig_wav_files):
        for wav in os.listdir(os.path.join(wav_file_path,orig_wav_file)):
            if wav[-3:]!='wav':
                pass
            else:
                signal, sr = librosa.load(os.path.join(wav_file_path,orig_wav_file,wav), fs)
                #signal = torch.tensor(signal).to(device)
                # signals.append( torch.tensor( process_func(signal[None,:], fs, embeddings=True)).type(torch.float32).to(device) )
                signals.append( process_func(signal[None,:], fs, embeddings=True)[0] )
    return

from preprocessing import *
print(torch.cuda.is_available())

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
signals=[]
get_signal_transformer(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav')
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

# signals = []
# label_cat = []
# signals.append(np.array([1.,2.,3.]))
# signals.append(np.array([1.,2.,3.]))
# signals.append(np.array([1.,2.,3.]))
# signals.append(np.array([1.,2.,3.]))
# signals.append(np.array([1.,2.,3.]))
# signals.append(np.array([1.,2.,3.]))
# signals.append(np.array([1.,2.,3.]))
# signals.append(np.array([1.,2.,3.]))
# signals.append(np.array([1.,2.,3.]))
# label_cat.append(np.array([1,0,0,0,0,0,0,0,0]))
# label_cat.append(np.array([1,0,0,0,0,0,0,0,0]))
# label_cat.append(np.array([1,0,0,0,0,0,0,0,0]))
# label_cat.append(np.array([1,0,0,0,0,0,0,0,0]))
# label_cat.append(np.array([1,0,0,0,0,0,0,0,0]))
# label_cat.append(np.array([1,0,0,0,0,0,0,0,0]))
# label_cat.append(np.array([1,0,0,0,0,0,0,0,0]))
# label_cat.append(np.array([1,0,0,0,0,0,0,0,0]))
# label_cat.append(np.array([1,0,0,0,0,0,0,0,0]))
X, X_test, y, y_test = train_test_split(signals, label_cat, test_size=0.1,random_state=0)
X_train, X_val,y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=0)
# y_train = torch.tensor(y_train).to(device)
# y_val = torch.tensor(y_val).to(device)
# y_test = torch.tensor(y_test).to(device)

from torch.utils.data import DataLoader, TensorDataset

# np.savetxt("D:\\pycharmProject\\FYP\\X_train.txt", X_train)
# np.savetxt("D:\\pycharmProject\\FYP\\X_test.txt", X_test)
# np.savetxt("D:\\pycharmProject\\FYP\\X_val.txt",X_val)
# np.savetxt("D:\\pycharmProject\\FYP\\y_train.txt", y_train)
# np.savetxt("D:\\pycharmProject\\FYP\\y_test.txt", y_test)
# np.savetxt("D:\\pycharmProject\\FYP\\y_val.txt", y_val)

dl_train = DataLoader(TensorDataset(torch.tensor(X_train).type(torch.float32).to(device),torch.tensor(y_train).type(torch.float32).to(device)),batch_size=32)
dl_test = DataLoader(TensorDataset(torch.tensor(X_test).type(torch.float32).to(device),torch.tensor(y_test).type(torch.float32).to(device)),batch_size=32)
dl_val = DataLoader(TensorDataset(torch.tensor(X_val).type(torch.float32).to(device),torch.tensor(y_val).type(torch.float32).to(device)),batch_size=32)



class Model(nn.Module):
    def __init__(self,input_size=1024, hidden_layer=[1024,512,128], output_size=9):
        super().__init__()
        self.activation = nn.ReLU()
        self.dense_layers = []
        self.num_hidden = len(hidden_layer)
        num_layer = len(hidden_layer)
        for i in range(num_layer):
            if i==0:
                input = input_size
            else:
                input = hidden_layer[i-1]
            self.dense_layers.append(nn.Linear(input, hidden_layer[i]).to(device))
        self.output_layer = nn.Linear(hidden_layer[-1],output_size)

    def forward(self,x):
        for layer in self.dense_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.output_layer(x)
        return x

def train(model, dl_train,loss_function, optimizer,  epochs=150, batch_size=32,path='D:\\pycharmProject\\FYP'):
    for i in trange(epochs):
        for step, data in enumerate(dl_train):
            x, y = data
            x=x.to(device)
            y=y.to(device)
            y_pred = model(x)
            loss = loss_function(y_pred,y)
            loss.backward()
            optimizer.step()
    torch.save(model,path)
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


print(f'signals[0].shape={signals[0].shape}')
model = Model(input_size=signals[0].shape[0], hidden_layer=[1024,512,128], output_size=9)
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model,dl_train,loss_function,optimizer,epochs=150, path='D:\\pycharmProject\\FYP\\transformer2.pth')
loss, error_rate = eval(model,dl_train,loss_function)
print(f'loss={loss},error rate={error_rate}')
