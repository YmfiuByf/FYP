from preprocessing import *
from DBN import *
from RBM import *

import librosa
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda
import matplotlib.pyplot as plt
import numpy as np
from transformers import Wav2Vec2Processor, get_scheduler
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence

def get_processed_signal(signals,  processor, fp, transformer, fs=16000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav_file_path = fp
    orig_wav_files = os.listdir(wav_file_path)
    for orig_wav_file in tqdm(orig_wav_files):
        for wav in os.listdir(os.path.join(wav_file_path,orig_wav_file)):
            if wav[-3:]!='wav':
                pass
            else:
                signal, sr = librosa.load(os.path.join(wav_file_path,orig_wav_file,wav), fs)
                signal = processor(signal[None,:], sampling_rate=fs)
                signal = signal['input_values'][0]
                signal = torch.from_numpy(signal)
                signal = transformer(signal)[0]
                signal.to(device)
                signals.append( signal )
    return

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

class Embed_Emo(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.2):
        super.__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(output_size)

    def forward(self, input_embedding):
        x =  input_embedding
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Model(nn.Module):
    def __init__(self, transformer, hidden_size, output_size, dropout=0.2):
        super.__init__()
        self.transformer = transformer
        self.classifier = Embed_Emo(hidden_size, output_size, dropout)

# load model from hub
device = 'cuda'
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
transformer = EmotionModel.from_pretrained(model_name)

# dummy signal
sampling_rate = 16000
# signal1 = np.zeros((1000), dtype=np.float32)
# signal2 = np.zeros((10000), dtype=np.float32)
# signal3 = np.zeros((100000), dtype=np.float32)
# c=[]
# c.append(signal1)
# c.append(signal2)
# c.append(signal3)
# d = pack_sequence(c, enforce_sorted=False)

#print(model)
#print(processor)
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
signals=[]
get_processed_signal(signals, processor=processor, fp='D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav',transformer= transformer, fs=16000)
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\sentences\\wav')
# get_signal(signals,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\sentences\\wav')
# print(len(signals))
# print(signals[0].size(),signals[1].size())

label_dim,label_cat=[],[]
get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\dialog\\EmoEvaluation')
# get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session2\\dialog\\EmoEvaluation')
# get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session3\\dialog\\EmoEvaluation')
# get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session4\\dialog\\EmoEvaluation')
# get_label_from_EmoEval(label_dim,label_cat,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\dialog\\EmoEvaluation')


X, X_test, y, y_test = train_test_split(signals, label_cat, test_size=0.1)
X_train, X_val,y_train, y_val = train_test_split(X,y,test_size=0.2)
y_train = torch.tensor(y_train).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs , batch_size = 100 , 4
num = ( len(X_train)// batch_size)
num_training_steps = num_epochs * num
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

def train_transformer(model, optimizer, loss_func, lr_scheduler, X_train, y_train, model_path, batch_size=4):
    model.train()
    for epoch in range(num_epochs):
        for i in range(num):
            y_pred = model(pad_sequence(X_train[batch_size*i: batch_size*i+batch_size]).transpose(1,0))[0]
            loss = loss_func(y_pred, y_train[batch_size*i: batch_size*i+batch_size,:])
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    torch.save(model,model_path)
    return
loss_func = nn.CrossEntropyLoss()
train_transformer(model, optimizer, loss_func, lr_scheduler, X_train, y_train, 'D:\\pycharmProject\\FYP\\wav2vec.pth')
