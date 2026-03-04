from preprocessing import *
import pandas as pd
import numpy as np
from tqdm import *
import librosa
import os
import warnings
from itertools import chain

torch.cuda.empty_cache()
def feature_IEMOCAP(n_mfcc,session=1):
    def librosa_mfcc(labels, flags,file_path,mfccs,n_mfcc=n_mfcc,fs=16000):
        warnings.filterwarnings("ignore")
        dict = {'neu': 1, 'sad': 2, 'fea': 3, 'fru': 4, 'ang': 5, 'sur': 6, 'dis': 7, 'hap': 8, 'exc': 9,'xxx': 0}
        def extract(line,labels,flags,i,path,mfccs):
            if 'Ses' not in line:
                return mfccs
            else:
                label = line[-28:-25]
                arr = 0
                th = line[line.rfind('S'):-29]
                pa = file_path[: file_path.rfind('dialog')]
                name = path[path.rfind('Ses'):path.rfind('.txt')]
                end_path = pa+'sentences\\wav\\'+name+'\\'+th+'.wav' #"D:\pycharmProject\FYP\IEMOCAP语料库\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F000.wav"
                signal, sr = librosa.load(end_path,sr=fs)

                if label in ['neu','sad','fea','fru','ang','sur','dis','hap','exc','xxx']:
                    arr = dict[label]
                else:
                    return mfccs

                assert fs == sr
                if n_mfcc > 0:
                    feature = np.transpose(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,win_length=400,n_fft=512,hop_length=160))
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
                    f0, voiced_flag, voiced_prob = librosa.pyin(signal, fmin=65, fmax=2093, sr=sr,win_length=400,hop_length=160)
                    # delta = librosa.feature.delta(f0, order=1)
                    # delta_delta = librosa.feature.delta(f0, order=2)
                    # fd = np.concatenate([f0,])
                    mfcc = pd.DataFrame(f0[:])
                    # print(mfcc.shape)
                    mfcc['flag'] = voiced_flag[:]
                    mfcc['prob'] = voiced_prob[:]

                mfcc['label'] = np.ones((len(mfcc), 1), dtype=int) * arr
                mfccs = pd.concat([mfccs, mfcc])
                return mfccs

        for file in tqdm(os.listdir(file_path)):
            if file[-3:]!='txt':
                continue
            path = os.path.join(file_path,file)
            with open(path, "r", encoding="utf-8") as f:
                k_list = savelist.read(f.read())
            for i in range(len(k_list)):
                mfccs = extract(k_list[i],labels,flags,i,path,mfccs)
        return mfccs


    def librosa_mfcc_train_test_split(labels, flags,file_path,mfccs,mfccs_test,n_mfcc=n_mfcc,fs=16000,test_size=0.2):
        warnings.filterwarnings("ignore")
        dict = {'neu': 1, 'sad': 2, 'fea': 3, 'fru': 4, 'ang': 5, 'sur': 6, 'dis': 7, 'hap': 8, 'exc': 9,'xxx': 0}
        def extract(line,labels,flags,i,path,mfccs,mfccs_test):
            if 'Ses' not in line:
                return mfccs,mfccs_test
            else:
                label = line[-28:-25]
                arr = 0
                th = line[line.rfind('S'):-29]
                pa = file_path[: file_path.rfind('dialog')]
                name = path[path.rfind('Ses'):path.rfind('.txt')]
                end_path = pa+'sentences\\wav\\'+name+'\\'+th+'.wav' #"D:\pycharmProject\FYP\IEMOCAP语料库\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F000.wav"
                signal, sr = librosa.load(end_path,sr=fs)

                if label in ['neu','sad','fea','fru','ang','sur','dis','hap','exc','xxx']:
                    arr = dict[label]
                else:
                    return mfccs,mfccs_test

                assert fs == sr
                if n_mfcc > 0:
                    feature = np.transpose(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,win_length=400,n_fft=512,hop_length=160))
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
                    f0, voiced_flag, voiced_prob = librosa.pyin(signal, fmin=65, fmax=2093, sr=sr,win_length=400,hop_length=160)
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
                if np.random.rand()>test_size:
                    mfccs = pd.concat([mfccs, mfcc])
                else:
                    mfccs_test = pd.concat([mfccs_test,mfcc])
                return mfccs,mfccs_test

        for file in tqdm(os.listdir(file_path)):
            if file[-3:]!='txt':
                continue
            path = os.path.join(file_path,file)
            with open(path, "r", encoding="utf-8") as f:
                k_list = savelist.read(f.read())
            for i in range(len(k_list)):
                mfccs,mfccs_test = extract(k_list[i],labels,flags,i,path,mfccs,mfccs_test)
        return mfccs,mfccs_test

    labels,flags,file_path = [], [], f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{session}\\dialog\\EmoEvaluation'
    # mfccs = pd.DataFrame()
    test_size=0.2
    mfccs,mfccs_test = pd.DataFrame(), pd.DataFrame()
    mfccs,mfccs_test = librosa_mfcc_train_test_split(labels,flags,file_path,mfccs,mfccs_test=mfccs_test,test_size=test_size)
    # print(mfccs.shape,mfccs_test.shape)
    return mfccs,mfccs_test

def feature_IEMOCAP(n_mfcc,session=1):
    def librosa_mfcc(labels, flags,file_path,mfccs,n_mfcc=n_mfcc,fs=16000):
        warnings.filterwarnings("ignore")
        dict = {'neu': 1, 'sad': 2, 'fea': 3, 'fru': 4, 'ang': 5, 'sur': 6, 'dis': 7, 'hap': 8, 'exc': 9,'xxx': 0}
        def extract(line,labels,flags,i,path,mfccs):
            if 'Ses' not in line:
                return mfccs
            else:
                label = line[-28:-25]
                arr = 0
                th = line[line.rfind('S'):-29]
                pa = file_path[: file_path.rfind('dialog')]
                name = path[path.rfind('Ses'):path.rfind('.txt')]
                end_path = pa+'sentences\\wav\\'+name+'\\'+th+'.wav' #"D:\pycharmProject\FYP\IEMOCAP语料库\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F000.wav"
                signal, sr = librosa.load(end_path,sr=fs)

                if label in ['neu','sad','fea','fru','ang','sur','dis','hap','exc','xxx']:
                    arr = dict[label]
                else:
                    return mfccs

                assert fs == sr
                if n_mfcc > 0:
                    feature = np.transpose(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,win_length=400,n_fft=512,hop_length=160))
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
                    f0, voiced_flag, voiced_prob = librosa.pyin(signal, fmin=65, fmax=2093, sr=sr,win_length=400,hop_length=160)
                    # delta = librosa.feature.delta(f0, order=1)
                    # delta_delta = librosa.feature.delta(f0, order=2)
                    # fd = np.concatenate([f0,])
                    mfcc = pd.DataFrame(f0[:])
                    # print(mfcc.shape)
                    mfcc['flag'] = voiced_flag[:]
                    mfcc['prob'] = voiced_prob[:]

                mfcc['label'] = np.ones((len(mfcc), 1), dtype=int) * arr
                mfccs = pd.concat([mfccs, mfcc])
                return mfccs

        for file in tqdm(os.listdir(file_path)):
            if file[-3:]!='txt':
                continue
            path = os.path.join(file_path,file)
            with open(path, "r", encoding="utf-8") as f:
                k_list = savelist.read(f.read())
            for i in range(len(k_list)):
                mfccs = extract(k_list[i],labels,flags,i,path,mfccs)
        return mfccs


    def librosa_mfcc_train_test_split(labels, flags,file_path,mfccs,mfccs_test,n_mfcc=n_mfcc,fs=16000,test_size=0.2):
        warnings.filterwarnings("ignore")
        dict = {'neu': 1, 'sad': 2, 'fea': 3, 'fru': 4, 'ang': 5, 'sur': 6, 'dis': 7, 'hap': 8, 'exc': 9,'xxx': 0}
        def extract(line,labels,flags,i,path,mfccs,mfccs_test):
            if 'Ses' not in line:
                return mfccs,mfccs_test
            else:
                label = line[-28:-25]
                arr = 0
                th = line[line.rfind('S'):-29]
                pa = file_path[: file_path.rfind('dialog')]
                name = path[path.rfind('Ses'):path.rfind('.txt')]
                end_path = pa+'sentences\\wav\\'+name+'\\'+th+'.wav' #"D:\pycharmProject\FYP\IEMOCAP语料库\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F000.wav"
                signal, sr = librosa.load(end_path,sr=fs)

                if label in ['neu','sad','fea','fru','ang','sur','dis','hap','exc','xxx']:
                    arr = dict[label]
                else:
                    return mfccs,mfccs_test

                assert fs == sr
                if n_mfcc > 0:
                    feature = np.transpose(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,win_length=400,n_fft=512,hop_length=160))
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
                    f0, voiced_flag, voiced_prob = librosa.pyin(signal, fmin=65, fmax=2093, sr=sr,win_length=400,hop_length=160)
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
                if np.random.rand()>test_size:
                    mfccs = pd.concat([mfccs, mfcc])
                else:
                    mfccs_test = pd.concat([mfccs_test,mfcc])
                return mfccs,mfccs_test

        for file in tqdm(os.listdir(file_path)):
            if file[-3:]!='txt':
                continue
            path = os.path.join(file_path,file)
            with open(path, "r", encoding="utf-8") as f:
                k_list = savelist.read(f.read())
            for i in range(len(k_list)):
                mfccs,mfccs_test = extract(k_list[i],labels,flags,i,path,mfccs,mfccs_test)
        return mfccs,mfccs_test

    labels,flags,file_path = [], [], f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{session}\\dialog\\EmoEvaluation'
    # mfccs = pd.DataFrame()
    test_size=0.2
    mfccs,mfccs_test = pd.DataFrame(), pd.DataFrame()
    mfccs,mfccs_test = librosa_mfcc_train_test_split(labels,flags,file_path,mfccs,mfccs_test=mfccs_test,test_size=test_size)
    # print(mfccs.shape,mfccs_test.shape)
    return mfccs,mfccs_test


def IEMOCAP_path(session=1):
    dict = {'neu': 1, 'sad': 2, 'fea': 3, 'fru': 4, 'ang': 5, 'sur': 6, 'dis': 7, 'hap': 8, 'exc': 9,'xxx': 0}
    file_path = 'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{session}\\dialog\\EmoEvaluation'
    def extract(line,i,path,mfccs):
        print('ex')
        if 'Ses' not in line:
            return mfccs
        else:
            label = line[-28:-25]
            arr = 0
            th = line[line.rfind('S'):-29]
            pa = file_path[: file_path.rfind('dialog')]
            name = path[path.rfind('Ses'):path.rfind('.txt')]
            end_path = pa+'sentences\\wav\\'+name+'\\'+th+'.wav' #"D:\pycharmProject\FYP\IEMOCAP语料库\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F000.wav"

            if label in ['neu','sad','fea','fru','ang','sur','dis','hap','exc','xxx']:
                arr = dict[label]
            else:
                return mfccs
            emotion = label
            emotion_intensity = 'normal'
            if 'F' in name:
                gender = 'female'
            else:
                gender = 'male'

            mfccs = mfccs.append({"Emotion": emotion,
                                "Emotion intensity": emotion_intensity,
                                "Gender": gender,
                                "Path": end_path
                                },
                               ignore_index=True
                               )
            return mfccs
        mfccs = pd.DataFrame(columns=['Emotion', 'Emotion intensity', 'Gender','Path'])

        for file in tqdm(os.listdir(file_path)):
            if file[-3:]!='txt':
                continue
            path = os.path.join(file_path,file)
            with open(path, "r", encoding="utf-8") as f:
                k_list = savelist.read(f.read())
            for i in range(len(k_list)):
                mfccs = extract(k_list[i],i,path,mfccs)
        mfccs.to_csv(rf"E:\FYP excel\IEMOCAP1_paths.csv")
        return mfccs



def IEMOCAP_paths(n_mfcc,session=1):
    def librosa_mfcc(labels, flags,file_path,mfccs,n_mfcc=n_mfcc,fs=16000):
        warnings.filterwarnings("ignore")
        dict = {'neu': 1, 'sad': 2, 'fea': 3, 'fru': 4, 'ang': 5, 'sur': 6, 'dis': 7, 'hap': 8, 'exc': 9,'xxx': 0}
        def extract(line,labels,flags,i,path,mfccs):
            if 'Ses' not in line:
                return mfccs
            else:
                label = line[-28:-25]
                arr = 0
                th = line[line.rfind('S'):-29]
                pa = file_path[: file_path.rfind('dialog')]
                name = path[path.rfind('Ses'):path.rfind('.txt')]
                end_path = pa+'sentences\\wav\\'+name+'\\'+th+'.wav' #"D:\pycharmProject\FYP\IEMOCAP语料库\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F000.wav"
                signal, sr = librosa.load(end_path,sr=fs)

                if label in ['neu','sad','fea','fru','ang','sur','dis','hap','exc','xxx']:
                    arr = dict[label]
                else:
                    return mfccs

                assert fs == sr
                if n_mfcc > 0:
                    feature = np.transpose(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,win_length=400,n_fft=512,hop_length=160))
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
                    f0, voiced_flag, voiced_prob = librosa.pyin(signal, fmin=65, fmax=2093, sr=sr,win_length=400,hop_length=160)
                    # delta = librosa.feature.delta(f0, order=1)
                    # delta_delta = librosa.feature.delta(f0, order=2)
                    # fd = np.concatenate([f0,])
                    mfcc = pd.DataFrame(f0[:])
                    # print(mfcc.shape)
                    mfcc['flag'] = voiced_flag[:]
                    mfcc['prob'] = voiced_prob[:]

                mfcc['label'] = np.ones((len(mfcc), 1), dtype=int) * arr
                mfccs = pd.concat([mfccs, mfcc])
                return mfccs

        for file in tqdm(os.listdir(file_path)):
            if file[-3:]!='txt':
                continue
            path = os.path.join(file_path,file)
            with open(path, "r", encoding="utf-8") as f:
                k_list = savelist.read(f.read())
            for i in range(len(k_list)):
                mfccs = extract(k_list[i],labels,flags,i,path,mfccs)
        return mfccs


    def librosa_mfcc_train_test_split(labels, flags,file_path,mfccs,mfccs_test,n_mfcc=n_mfcc,fs=16000,test_size=0.2):
        warnings.filterwarnings("ignore")
        dict = {'neu':0,'sad':1,'hap':2,'fru':3,'ang':4,'exc':5}
        def extract(line,labels,flags,i,path,mfccs,mfccs_test):
            if 'Ses' not in line:
                return mfccs,mfccs_test
            else:
                label = line[-28:-25]
                arr = 0
                th = line[line.rfind('S'):-29]
                pa = file_path[: file_path.rfind('dialog')]
                name = path[path.rfind('Ses'):path.rfind('.txt')]
                end_path = pa+'sentences\\wav\\'+name+'\\'+th+'.wav' #"D:\pycharmProject\FYP\IEMOCAP语料库\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F000.wav"
                signal, sr = librosa.load(end_path,sr=fs)

                if label in dict.keys():
                    arr = dict[label]
                else:
                    return mfccs,mfccs_test

                assert fs == sr

                emotion = arr
                emotion_intensity = 'normal'
                if 'F' in name:
                    gender = 'female'
                else:
                    gender = 'male'

                mfccs = mfccs.append({"Emotion": emotion,
                                      "Emotion intensity": emotion_intensity,
                                      "Gender": gender,
                                      "Path": end_path
                                      },
                                     ignore_index=True
                                     )
                return mfccs,mfccs_test

        for file in tqdm(os.listdir(file_path)):
            if file[-3:]!='txt':
                continue
            path = os.path.join(file_path,file)
            with open(path, "r", encoding="utf-8") as f:
                k_list = savelist.read(f.read())
            for i in range(len(k_list)):
                mfccs,mfccs_test = extract(k_list[i],labels,flags,i,path,mfccs,mfccs_test)
        mfccs.to_csv(rf"E:\FYP excel\IEMOCAP1_paths.csv")
        return mfccs,mfccs_test

    labels,flags,file_path = [], [], f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{session}\\dialog\\EmoEvaluation'
    # mfccs = pd.DataFrame()
    test_size=0.2
    mfccs,mfccs_test = pd.DataFrame(), pd.DataFrame()
    mfccs,mfccs_test = librosa_mfcc_train_test_split(labels,flags,file_path,mfccs,mfccs_test=mfccs_test,test_size=test_size)
    # print(mfccs.shape,mfccs_test.shape)
    return mfccs,mfccs_test


if __name__=='__main__':
    session = 1

    # IEMOCAP_paths(0)


    n_mfcc = 80
    test_size=0.2
    mfccs,mfccs_test = feature_IEMOCAP(n_mfcc=n_mfcc, session=session)
    mfccs.to_csv(rf"E:\FYP excel\IEMOCAP{session}_MFCC{n_mfcc}_train={1 - test_size}.csv")
    mfccs_test.to_csv(rf"E:\FYP excel\IEMOCAP{session}_MFCC{n_mfcc}_test={test_size}.csv")


    # for n_mfcc in range(10,41):
    #     print(f'n_mfcc={n_mfcc}')
    #     main(n_mfcc=n_mfcc,session=1)

