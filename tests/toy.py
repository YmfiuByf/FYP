device = 'cuda'
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import warnings
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from LSTM import LSTM
from preprocessing import *
import os
from tqdm import *
import librosa
import teager_py
dropout = 0.5
from torchvision.models import inception
import torch
output_size=6
conv = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
# x = torch.arange(24).reshape(1,2,3,4)
# x_ = x.clone()
# x = torch.flatten(x,start_dim=1,end_dim=2)
# print(x,x_)
# x = x.reshape(x_.size())
# print(torch.unique(x==x_))
x = torch.ones(2,4)*2
print(torch.softmax(x,dim=1))