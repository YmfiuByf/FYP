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
import sklearn
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import joblib
import sys


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

    # Disable
    def blockPrint():
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def enablePrint():
        sys.stdout = sys.__stdout__

    blockPrint()

    def signal_labelling(signals, clf):
        ret = []
        for signal in tqdm(signals):
            frame_label = clf.predict(signal)
            label, cnt = np.unique(frame_label, return_counts=True)
            idx = np.argmax(cnt)
            label = label[idx]
            ret.append(label)
        return np.array(ret)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    flags2_test,label_cat_test = [], []
    get_label_4cat(label_cat_test,flags2_test,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\dialog\\EmoEvaluation')
    signals_test,flags1_test,length_test = [],[],[]
    get_mfcc_f0(signals_test,flags1_test,length_test,'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session5\\sentences\\wav')

    def four_emo(signals, flags, length, label_cat):
        flags = np.array(flags)
        length = np.array(length)
        signals = np.array(signals)
        label_cat = np.array(label_cat)
        idx = flags == 1
        signals = signals[idx]
        length = length[idx]
        label_cat = label_cat[idx]
        return signals, length, label_cat


    flags1_test = np.array(flags1_test)
    flags2_test = np.array(flags2_test)
    flags_test = flags1_test.astype(bool) & flags2_test.astype(bool)
    signals_test,length_test, label_cat_test = four_emo(signals_test,flags_test,length_test, label_cat_test)

    y = np.repeat(label_cat_test, length_test, axis=0)
    X = []
    original_signal = signals_test
    for signal in signals_test:
        for feature in signal:
            X.append(feature)
    X = np.array(X)
    print(len(X), len(y))

    enablePrint()

    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    signals=[]
    get_original_signal(signals,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{s}\\sentences\\wav')

    label_cat,flags=[],[]
    get_label_4cat(label_cat,flags,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{s}\\dialog\\EmoEvaluation')


    flags = np.array(flags)
    signals = np.array(signals)
    # label_cat = np.array(label_cat)
    idx = flags == 1
    signals = signals[idx]
    # label_cat = label_cat[idx]
    signals = signals.tolist()

    X_test = signals
    y_test = label_cat

    def signal_labelling2(signals, clf,model):
        y_preds = []
        y_gts = []
        for i in trange(len(signals)):
            signal = signals_test[i]
            prob1 = clf.predict_proba(signal)
            prob1 = np.mean(prob1,axis=0)
            x = X_test[i][None,:]
            gt = np.array(y_test[i][None, :])
            y_pred = process_func(x, sampling_rate=16000)
            prob2 = y_pred.detach().cpu().numpy()
            w1 = np.array([0.7,0.5,0.3,0.6])
            w2 = 1 - w1
            prob = w1*prob1 + w2*prob2
            label = np.argmax(prob)
            y_preds.append(label)
            y_gts.append(gt)
        return np.array(y_preds), np.array(y_gts)

    clf = joblib.load(r"D:\pycharmProject\FYP\mfccF0_IMB_HistGradientBoostingClassifier.plk")
    y_preds,y_gts = signal_labelling2(signals,clf,model)
    accuracy = metrics.accuracy_score(y_gts,y_preds)
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
