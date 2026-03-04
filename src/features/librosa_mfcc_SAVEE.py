from preprocessing import *
import pandas as pd
import numpy as np
from tqdm import *
import librosa
import os
import warnings
from itertools import chain
warnings.filterwarnings('ignore')


def librosa_SAVEE(n_mfcc=15, fs=16000,test_size=0.2):
    root_path = r"E:\SAVEE\AudioData"
    dict = {'n': 1, 'f': 2, 'a': 3, 'd': 4,
            'h': 5}  # num2label3 = ['neural','fear','anger','disgust','happiness','sad','surprise']
    mfccs = pd.DataFrame()
    mfccs_test = pd.DataFrame()
    for file in os.listdir(root_path):
        if '.' in file:
            continue
        person_path = os.path.join(root_path, file)
        for wav_path in tqdm(os.listdir(person_path)):
            wav = os.path.join(person_path, wav_path)

            signal, sr = librosa.load(wav, sr=fs)

            if wav_path[0] == 's':
                if wav_path[1] == 'a':
                    arr = 6
                elif wav_path[1] == 'u':
                    arr = 7
            else:
                arr = dict[wav_path[0]]

            assert fs == sr
            if n_mfcc>0:
                feature = np.transpose(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,win_length=400,hop_length=160,n_fft=512))
                delta = librosa.feature.delta(feature, order=1)
                delta_delta = librosa.feature.delta(feature, order=2)
                # print(feature.shape,delta.shape,delta_delta.shape)
                fd = np.concatenate([feature, delta], axis=1)
                fdd = np.concatenate([fd, delta_delta], axis=1)
                # f0 = librosa.pyin(signal,fmin=65,fmax=2093,sr=sr)[0]
                # fdd = np.concatenate([fdd,f0[:,None]],axis=1)
                assert fdd.shape[1] == 3 * n_mfcc
                mfcc = pd.DataFrame(fdd)
            else:
                # cnt = len(mfccs) + len(mfccs_test)
                # np.random.seed(cnt)
                # if np.random.rand() > test_size:
                #     feature = np.transpose(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,win_length=400,hop_length=160,n_fft=512))
                #     mfcc = pd.DataFrame(feature)
                #
                # else:
                f0, voiced_flag, voiced_prob = librosa.pyin(signal,fmin=65,fmax=2093,sr=sr,win_length=400,hop_length=160)
                # delta = librosa.feature.delta(f0, order=1)
                # delta_delta = librosa.feature.delta(f0, order=2)
                # fd = np.concatenate([f0,])
                mfcc = pd.DataFrame(f0[:])
                # print(mfcc.shape)
                mfcc['flag'] = voiced_flag[:]
                mfcc['prob'] = voiced_prob[:]


            mfcc['label'] = np.ones((len(mfcc), 1), dtype=int) * arr
            cnt = len(mfccs) + len(mfccs_test)
            np.random.seed(cnt)
            if np.random.rand() > test_size:
                mfccs = pd.concat([mfccs, mfcc])
            else:
                mfccs_test = pd.concat([mfccs_test, mfcc])
    return mfccs,mfccs_test

def SAVEE_path():
    data = pd.DataFrame(columns=['Emotion', 'Emotion intensity', 'Gender', 'Path'])
    root_path = r"E:\SAVEE\AudioData"
    dict = {'n': 1, 'f': 2, 'a': 3, 'd': 4,
            'h': 5}  # num2label3 = ['neural','fear','anger','disgust','happiness','sad','surprise']
    EMOTIONS2 = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'neural', 5: 'sad', 6: 'surprise'}
    for file in os.listdir(root_path):
        if '.' in file:
            continue
        person_path = os.path.join(root_path, file)
        for wav_path in tqdm(os.listdir(person_path)):
            wav = os.path.join(person_path, wav_path)
            file_path = wav
            if wav_path[0] == 's':
                if wav_path[1] == 'a':
                    arr = 6
                elif wav_path[1] == 'u':
                    arr = 7
            else:
                arr = dict[wav_path[0]]

            gender = 'male'
            emotion_intensity = 'normal'
            emotion = arr-1 #EMOTIONS2[arr-1]
            data = data.append({"Emotion": emotion,
                                "Emotion intensity": emotion_intensity,
                                "Gender": gender,
                                "Path": file_path
                                },
                               ignore_index=True
                               )
        data.to_csv(rf"E:\FYP excel\SAVEE_paths.csv")
    return data

if __name__=='__main__':
    SAVEE_path()

    # n_mfcc = 57
    # test_size = 0.2
    # mfccs,mfccs_test = librosa_SAVEE(n_mfcc=n_mfcc,test_size=test_size)
    # mfccs.to_csv(rf"E:\FYP excel\SAVEE_MFCC{n_mfcc}_train={1 - test_size}.csv")
    # mfccs_test.to_csv(rf"E:\FYP excel\SAVEE_MFCC{n_mfcc}_test={test_size}.csv")

    # for n_mfcc in chain(range(10,20,2),range(20,40,3),range(40,101,5)):
    #     print(f'n_mfcc={n_mfcc}')
    #     mfccs,mfccs_test = librosa_SAVEE(n_mfcc=n_mfcc,test_size=test_size)
    #     mfccs.to_csv(rf"E:\FYP excel\SAVEE_MFCC{n_mfcc}_train={1 - test_size}.csv")
    #     mfccs_test.to_csv(rf"E:\FYP excel\SAVEE_MFCC{n_mfcc}_test={test_size}.csv")