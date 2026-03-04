import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import librosa
from scipy.io import wavfile
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoConfig

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






# def process_func(
#     x: np.ndarray,
#     sampling_rate: int,
#     embeddings: bool = False,
# ) -> np.ndarray:
#     r"""Predict emotions or extract embeddings from raw audio signal.
#     :rtype: object
#     """
#
#     # run through processor to normalize signal
#     # always returns a batch, so we just get the first entry
#     # then we put it on the device
#     y = processor(x, sampling_rate=sampling_rate)
#     y = y['input_values'][0]
#     y = torch.from_numpy(y).to(device)
#
#     # run through model
#     with torch.no_grad():
#         y = model(y)[0 if embeddings else 1]
#
#     # convert to numpy
#     y = y.detach().cpu().numpy()
#
#     return y

#if __name__=='main':
def test():

    model_checkpoint='facebook/wav2vec2-large-robust'
    batch_size=32
    config = AutoConfig.from_pretrained(model_checkpoint)

    tokenizer_type = config.model_type if config.tokenizer_class is None else None
    config = config if config.tokenizer_class is not None else None
    tokenizer = AutoTokenizer.from_pretrained(
        "./",
        config=config,
        tokenizer_type=tokenizer_type,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    # load model from hub
    device = 'cpu'
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name)
    print(processor)
    print(model)

    # dummy signal
    sampling_rate = 16000
    fs_1m, signal = wavfile.read("D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav\\Ses01F_impro01\\Ses01F_impro01_F000.wav")
    print(f'fs_1m={fs_1m},{type(data_1m)}')
    #signal, sr = librosa.load("D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav\\Ses01F_impro01\\Ses01F_impro01_F000.wav", 16000)
    # signal1, sr = librosa.load(
    #     "D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav\\Ses01F_impro01\\Ses01F_impro01_F001.wav",
    #     16000)
    # tmp=[]
    # tmp.append(signal)
    # tmp.append(signal1)
    # signalA=np.array(tmp)
    signal = np.array(signal)
    arr1=processor(signal, sampling_rate=sampling_rate)
    # print(arr1.shape)
    #  Arousal    dominance valence
    # [[0.5460759 0.6062269 0.4043165]]
    signal2, sr = librosa.load(
        "D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session1\\sentences\\wav\\Ses01F_impro01\\Ses01F_impro01_F001.wav",
        16000)
    signal2 = np.array(signal2)
    arr2=processor(signal2, sampling_rate=sampling_rate)
    #print(arr2['input_values'][0].shape,arr1['input_values'][0].shape)
    #arr2['input_values'].append(arr1['input_values'])
    print(arr2['attention_mask'][0].shape, arr1['attention_mask'][0].shape)

    batch = processor.pad(
        arr2,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
    )
    print(batch['input_values'][0].shape)
    print(type(arr2),len(arr2['input_values']))
    # Pooled hidden states of last transformer layer
    # [[-0.00752167  0.0065819  -0.00746339 ...  0.00663631  0.00848747
    #   0.00599209]]

