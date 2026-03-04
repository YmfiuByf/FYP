from __future__ import division
from scipy.fftpack import dct

file_path = ''

# 1.sample  2.preemphasis  3.fft
import decimal
import os
import librosa
import numpy as np
import math
import logging
from tqdm import tqdm
import torch
import Signal_Analysis.features.signal as sig
import scipy
import python_speech_features
import matplotlib.pyplot as plt
import pyttsx3
from pathlib import Path
import audiofile
import opensmile
import pandas as pd

engine = pyttsx3.init() # object creation

import math
def bilinear_interpolation(image, y, x):
    height = image.shape[0]
    width = image.shape[1]

    x1 = max(min(math.floor(x), width - 1), 0)
    y1 = max(min(math.floor(y), height - 1), 0)
    x2 = max(min(math.ceil(x), width - 1), 0)
    y2 = max(min(math.ceil(y), height - 1), 0)

    a = float(image[y1, x1])
    b = float(image[y2, x1])
    c = float(image[y1, x2])
    d = float(image[y2, x2])

    dx = x - x1
    dy = y - y1

    new_pixel = a * (1 - dx) * (1 - dy)
    new_pixel += b * dy * (1 - dx)
    new_pixel += c * dx * (1 - dy)
    new_pixel += d * dx * dy
    return round(new_pixel)


def resize(image, new_height, new_width):
    new_image = np.zeros((new_height, new_width), image.dtype)  # new_image = [[0 for _ in range(new_width)] for _ in range(new_height)]

    orig_height = image.shape[0]
    orig_width = image.shape[1]

    # Compute center column and center row
    x_orig_center = (orig_width-1) / 2
    y_orig_center = (orig_height-1) / 2

    # Compute center of resized image
    x_scaled_center = (new_width-1) / 2
    y_scaled_center = (new_height-1) / 2

    # Compute the scale in both axes
    scale_x = orig_width / new_width;
    scale_y = orig_height / new_height;

    for y in range(new_height):
        for x in range(new_width):
            x_ = (x - x_scaled_center) * scale_x + x_orig_center
            y_ = (y - y_scaled_center) * scale_y + y_orig_center

            new_image[y, x] = bilinear_interpolation(image, y_, x_)

    return new_image

def resize_3d(image,new_height,new_width):
    return np.concatenate( [np.concatenate([resize(image[:,:,0],new_height,new_width),resize(image[:,:,1],new_height,new_width)],axis=-1),resize(image[:,:,2],new_height,new_width)],axis=-1 ).reshape((new_height,new_width,3),order='F')

def text2speech(text,gender,file_name):
    """ RATE"""
    rate = engine.getProperty('rate')   # getting details of current speaking rate                     #printing current voice rate
    engine.setProperty('rate', 125)     # setting up new voice rate

    """VOLUME"""
    volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)                     #printing current volume level
    engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1

    """VOICE"""
    voices = engine.getProperty('voices')       #getting details of current voice
    #engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
    engine.setProperty('voice', voices[gender].id)   #changing index, changes voices. 1 for female
    engine.save_to_file(text, file_name)
    engine.runAndWait()
    return
# # default 16000Hz, 25ms<=>400 samples, step=10ms=160 samples, rectangular window
# def framesig(signal, frame_len=400, frame_step=160, winfunc=lambda x: np.ones((x,)) ):
#     slen = len(signal)
#     total_num = (slen-1)//frame_step
#     padsignal = np.concatenate((signal, np.zeros((total_num*frame_step+frame_len-slen,))))
#     win = winfunc(frame_len)
#
#     shape = padsignal.shape[:-1]+(total_num+1 , frame_len)  # [num of signal, num of frame, frame length]
#     strides = (padsignal.strides[0],) + (padsignal.strides[-1]*frame_step,) + (padsignal.strides[-1],)
#     frames = np.lib.stride_tricks.as_strided(padsignal, shape=shape, strides=strides)
#     return frames * win

def my_plot(*images):
    num = len(images)
    for i in range(num):
        plt.subplot(num,1,i+1).plot(images[i])
    plt.show()


def compute_F0(signal, rate=16000, kwargs={'accurate': 0, 'min_pitch': 75, 'pulse': True}):
    est_val0 = sig.get_F_0(signal, rate, **kwargs)
    contour = np.array((est_val0[1]))
    contour = 1.0 / contour
    return contour, est_val0[0]


def compute_voiced_time(signal, rate=16000, kwargs={'accurate': 0, 'min_pitch': 75, 'pulse': True}):
    est_val0 = np.array((sig.get_F_0(signal, rate, **kwargs)[2])) * 16000
    est_val0 = est_val0.astype(np.int32)
    return est_val0

def framesig(signal, frame_len=400, frame_step=160, winfunc=lambda shape: np.ones(shape) ):
    return frames_sig(signal[None,], frame_len=400, frame_step=160, winfunc=lambda num_frame, frame_len: np.ones((num_frame, frame_len)) )[0]

# this is used for signal array, framesig is for one single signal
def frames_sig(signals, frame_len=400, frame_step=160, winfunc=lambda num_frame, frame_len: np.ones((num_frame, frame_len)) ):
    sp = signals.shape
    slen = sp[1]
    batch_size = sp[0]
    total_num = (slen-1)//frame_step
    padsignal = np.concatenate((signals, np.zeros((batch_size, total_num*frame_step+frame_len-slen) )  ) ,axis=1)
    shape = padsignal.shape[:-1]+(total_num+1 , frame_len)  # [num of signal, num of frame, frame length]
    win = winfunc(total_num+1, frame_len)
    strides = (padsignal.strides[0],) + (padsignal.strides[-1]*frame_step,) + (padsignal.strides[-1],)
    frames = np.lib.stride_tricks.as_strided(padsignal, shape=shape, strides=strides)
    return frames * win



# energy in high frequency range is low, so increase them first
def preemphasis(signal, coeff=0.95):
    return np.append(signal[:,0].reshape(-1,1) , signal[:,1:] - coeff * signal[:,:-1] , axis=1)

# power spectrum   512 point fft
def powspec(frames, NFFT=512):
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))
# fft magnitude
def magspec(frames, NFFT=512):
    complex_spec = np.fft.rfft(frames, n=NFFT, axis=-1)
    return np.absolute(complex_spec)

# filter bank  fb=2 dims  [num of filter, 257=nfft/2+1]

def get_fb(num=26, lowfreq=300, highfreq=8000, fs=16000, nfft=512):
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melfb = np.linspace(lowmel, highmel, num+2)
    hzfb = mel2hz(melfb)
    bin = np.floor( (nfft+1)*hzfb/fs  ).astype(int)
    fb = np.zeros( (num, nfft//2+1 ) )
    for i in range(0, num):
        for j in range(bin[i], bin[i+1]):
            fb[i][j] = (j-bin[i])/(bin[i+1]-bin[i])
        for j in range(bin[i+1], bin[i+2]):
            fb[i][j] = (bin[i+2]-j)/(bin[i+2]-bin[i])
    return fb


# conver hz to mel, vice versa
def mel2hz(f_mel):
    return 700 * (10 ** (f_mel / 2595.0) - 1)

def hz2mel(f_hz):
    return 2595 * np.log10(1+f_hz/700.)

# improve the recog effect in noisy env
def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        _,nframes,ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2.)*np.sin(np.pi*n/L)
        return cepstra*lift
    else:
        # values of L <= 0, do nothing
        return cepstra


def get_label_from_txt(label_dim,label_cat,file_path):
    fp = open(file_path)
    cnt=0.
    cat=np.zeros((9))
    dim=np.zeros((3))
    last = ''
    for line in fp:
        if line[0]=='C':
            cnt+=1.
            if 'Sadness' in line:
                cat[0]+=1.
            if 'Fear' in line:
                cat[1]+=1. # ['neu','sad','fea','fru','ang','sur','dis','hap','exc']
            if 'Neutral' in line:
                cat[2]+=1.
            if 'Frustration' in line:
                cat[3]+=1.
            if 'Anger' in line:
                cat[4]+=1.
            if 'Surprise' in line:
                cat[5]+=1.
            if 'Disgust' in line:
                cat[6]+=1.
            if 'Happiness' in line:
                cat[7]+=1.
            if 'Excited' in line:
                cat[8]+=1.

        elif line[0]=='[' :
            cnt=0.
            cat=np.zeros((9))
            dim=np.zeros((3))
            dim[2]=float(line[-24:-18])       # V
            dim[0]=float(line[-16:-10])       # A
            dim[1]=float(line[-8:-2])           # D
            dim = dim/5.
            label_dim.append(dim)
        elif line[0]=='A'and last=='C':
            idx = np.argmax(cat)
            cat = np.zeros((9))
            cat[idx] = 1.
            # cat=cat/cnt

            label_cat.append(cat)
        last=line[0]
    return

def get_label_from_EmoEval(label_dim,label_cat,file_path):
    lst = os.listdir(file_path)
    for file in lst:
        if file[-3:] !='txt':
            pass
        else:
            path = os.path.join(file_path,file)
            get_label_from_txt(label_dim,label_cat,path)
    return

def get_label_from_EmoEval2(label_dim,label_cat,label_cat2,file_path):
    lst = os.listdir(file_path)
    for file in lst:
        if file[-3:] !='txt':
            pass
        else:
            path = os.path.join(file_path,file)
            get_label_from_txt2(label_dim,label_cat,label_cat2,path)
    return

def get_label_from_txt2(label_dim,label_cat,lable_cat2,file_path):
    fp = open(file_path)
    cnt=0.
    cat=np.zeros((9))
    dim=np.zeros((3))
    cat2 = np.ones((3))*0.3
    last = ''
    for line in fp:
        if line[0]=='C':
            cnt+=1.
            if 'Sadness' in line:  # neg:0,1,3,4,6  pos:7,8  neu:2,5
                cat[0]+=1.
            if 'Fear' in line:
                cat[1]+=1.
            if 'Neutral' in line:
                cat[2]+=1.
            if 'Frustration' in line:
                cat[3]+=1.
            if 'Anger' in line:
                cat[4]+=1.
            if 'Surprise' in line:
                cat[5]+=1.
            if 'Disgust' in line:
                cat[6]+=1.
            if 'Happiness' in line:
                cat[7]+=1.
            if 'Excited' in line:
                cat[8]+=1.

        elif line[0]=='[' :
            cnt=0.
            cat2 = np.ones((3))*0.3
            cat=np.zeros((9))
            dim=np.zeros((3))
            dim[2]=float(line[-24:-18])       # V
            dim[0]=float(line[-16:-10])       # A
            dim[1]=float(line[-8:-2])           # D
            dim = dim/5.
            label_dim.append(dim)
        elif line[0]=='A'and last=='C':
            idx = np.argmax(cat)
            cat = np.zeros((9))
            cat[idx] = 1.
            if idx in [7,8]:
                cat2[0]+=0.1
            elif idx in [2,5]:
                cat2[1]+=0.1
            elif idx in [0,1,3,4,6]:
                cat2[2]+=0.1
            lable_cat2.append(cat2)
            cat2 = np.ones((3))*0.3
            # cat=cat/cnt

            label_cat.append(cat)
        last=line[0]
    return
# step of mfcc:
# a.preprocessing: 1.sampling=>frames  2.preemphasis  3.filter
# b. 1.fft  2.melbank  3.



def mfccs(signals, fs=16000, len_sec=25, step_sec=10, nfb=26, nfft=512, numcep=13, ceplifter=22, appendEnergy=True):
    frame_len = int(len_sec * fs / 1000)
    frame_step = int(step_sec * fs / 1000)
    signal = preemphasis(signals, 0.95)

    def winfunc(num, frame_len):
        win = np.arange(frame_len)
        win = 0.54 - 0.46 * np.cos(2 * np.pi * win / (frame_len - 1))
        ret = np.ones((num, frame_len))
        return ret * win

    frames = frames_sig(signal, frame_len, frame_step, winfunc)  # 3 dims
    fb = get_fb(num=26, lowfreq=300, highfreq=8000, fs=16000, nfft=512)
    power_spec = powspec(frames, nfft)
    energy = np.sum(power_spec, axis=-1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    feat = np.dot(power_spec[:, :], fb.T)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)

    feat = np.log(feat)
    feat = dct(feat, type=2, axis=-1, norm='ortho')[:, :, :numcep]
    feat = lifter(feat, ceplifter)
    if appendEnergy:
        feat[:, :, 0] = np.log(energy)
    return feat   # [batch_size, num_frame, 13]  随信号长度改变而变化

def mfcc(signals, fs=16000, len_sec=25, step_sec=10, nfb=26, nfft=512, numcep=13, ceplifter=22, appendEnergy=True):
    frame_len = int(len_sec * fs / 1000)
    frame_step = int(step_sec * fs / 1000)
    signal = preemphasis(signals, 0.95)

    def winfunc(num, frame_len):
        win = np.arange(frame_len)
        win = 0.54 - 0.46 * np.cos(2 * np.pi * win / (frame_len - 1))
        ret = np.ones((num, frame_len))
        return ret * win

    frames = frames_sig(signal, frame_len, frame_step, winfunc)  # 3 dims
    fb = get_fb(num=nfb, lowfreq=300, highfreq=8000, fs=fs, nfft=nfft)
    power_spec = powspec(frames, nfft)
    energy = np.sum(power_spec, axis=-1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    feat = np.dot(power_spec[:, :], fb.T)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)

    feat = np.log(feat)
    feat = dct(feat, type=2, axis=-1, norm='ortho')[:, :, :numcep]
    feat = lifter(feat, ceplifter)
    if appendEnergy:
        feat[:, :, 0] = np.log(energy)
    return feat  #[ num_frame, 13 ]  随信号长度改变而变化

def deltas(x, n=2): # [len, 13]
    length = x.shape[0]
    dim = x.shape[1]
    c_r2 = x[2:]
    c_r2 = np.concatenate([c_r2,np.zeros([2,dim])],axis=0)
    c_r1 = x[1:]
    c_r1 = np.concatenate([c_r1,np.zeros([1,dim])],axis=0)
    c_l1 = x[:length-1]
    c_l1 = np.concatenate([np.zeros([1,dim]),c_l1],axis=0)
    c_l2 = x[:length-2]
    c_l2 = np.concatenate([np.zeros([2,dim]),c_l2],axis=0)
    num = 2*(c_r2-c_l2)+1*(c_r1-c_l1)
    denum = 2.*(1+4)
    ret = num/denum
    return ret

def mfcc_deltas(signal, delta=0):
    ret = mfcc(signal[None,])[0]
    delta1 = deltas(ret)
    delta2 = deltas(delta1)
    if delta>0:
        ret = np.concatenate([ret,delta1],axis=1)
        if delta>1:
            ret = np.concatenate([ret,delta2],axis=1)
    return ret

def mfcc_f0(signal, fs=16000, len_sec=25, step_sec=10, nfb=26, nfft=512, numcep=13, ceplifter=22, appendEnergy=True):
    rate = fs
    kwargs = { 'accurate'  : 0, 'min_pitch' : 75 ,'pulse':True}
    frame_len = int(len_sec * fs / 1000)
    frame_step = int(step_sec * fs / 1000)
#     signal = preemphasis(signals, 0.95)

    def winfunc(num, frame_len):
        win = np.arange(frame_len)
        win = 0.54 - 0.46 * np.cos(2 * np.pi * win / (frame_len - 1))
        ret = np.ones((num, frame_len))
        return ret * win
    est_val = sig.get_F_0( signal, rate, **kwargs,time_step=0 )
    f0 = np.array(est_val[1])[None,:,None]
    f0 = 1.0/f0
    est_val0 = np.array(est_val[2])*16000
    if len(est_val0.shape)==2:
        shape = est_val0.shape
        if shape[0]<=1:
            flag = 0
            feat = np.zeros((1,2,14))
        else:
            flag = 1
            est_val0 = est_val0.astype(np.int)
            frames = np.zeros((1,len(est_val0),est_val0[0,1]-est_val0[0,0]))
            for i in range(len(est_val0)):
                frames[0,i]=signal[est_val0[i][0]:est_val0[i][1]]
        #     frames = frames_sig(signal, frame_len, frame_step, winfunc)  # 3 dims
            fb = get_fb(num=26, lowfreq=300, highfreq=8000, fs=16000, nfft=512)
            power_spec = powspec(frames, nfft)
            energy = np.sum(power_spec, axis=-1)
            energy = np.where(energy == 0, np.finfo(float).eps, energy)
            feat = np.dot(power_spec[:, :], fb.T)
            feat = np.where(feat == 0, np.finfo(float).eps, feat)
            feat = np.log(feat)
            feat = dct(feat, type=2, axis=-1, norm='ortho')[:, :, :numcep]
            feat = lifter(feat, ceplifter)
            if appendEnergy:
                feat[:, :, 0] = np.log(energy)
            feat = np.concatenate((feat,f0),axis=-1)
    else:
        flag = 0
        feat = np.zeros((1,2,14))
    return feat,flag  #[ num_signal, num_frame, 14 ]  随信号长度改变而变化

def another_mfcc_delta(signal, fs=16000, len_sec=25, step_sec=10, nfb=26, nfft=512, numcep=13, ceplifter=22, appendEnergy=True):
    rate = fs
    kwargs = { 'accurate'  : 0, 'min_pitch' : 75 ,'pulse':True}
    frame_len = int(len_sec * fs / 1000)
    frame_step = int(step_sec * fs / 1000)
#     signal = preemphasis(signals, 0.95)

    def winfunc(num, frame_len):
        win = np.arange(frame_len)
        win = 0.54 - 0.46 * np.cos(2 * np.pi * win / (frame_len - 1))
        ret = np.ones((num, frame_len))
        return ret * win
    est_val = sig.get_F_0( signal, rate, **kwargs,time_step=0 )
    f0 = np.array(est_val[1])[None,:,None]
    f0 = 1.0/f0
    est_val0 = np.array(est_val[2])*16000
    if len(est_val0.shape)==2:
        shape = est_val0.shape
        if shape[0]<=1:
            flag = 0
            feat = np.zeros((1,2,13))
        else:
            flag = 1
            est_val0 = est_val0.astype(np.int)
            frames = np.zeros((1,len(est_val0),est_val0[0,1]-est_val0[0,0]))
            for i in range(len(est_val0)):
                frames[0,i]=signal[est_val0[i][0]:est_val0[i][1]]
        #     frames = frames_sig(signal, frame_len, frame_step, winfunc)  # 3 dims
            fb = get_fb(num=26, lowfreq=300, highfreq=8000, fs=16000, nfft=512)
            power_spec = powspec(frames, nfft)
            energy = np.sum(power_spec, axis=-1)
            energy = np.where(energy == 0, np.finfo(float).eps, energy)
            feat = np.dot(power_spec[:, :], fb.T)
            feat = np.where(feat == 0, np.finfo(float).eps, feat)
            feat = np.log(feat)
            feat = dct(feat, type=2, axis=-1, norm='ortho')[:, :, :numcep]
            feat = lifter(feat, ceplifter)
            if appendEnergy:
                feat[:, :, 0] = np.log(energy)
    else:
        flag = 0
        feat = np.zeros((1,2,13))
    return feat,flag  #[ num_signal, num_frame, 14 ]  随信号长度改变而变化


def feature_final(signal, fs=16000, len_sec=25, step_sec=10, m=13, nfft=512, numcep=13, ceplifter=22, appendEnergy=True,
                  delta=0, fundamental_frequency=True):
    assert delta in [0,1,2]
    nfb = m * 2
    feature_length = m * (delta + 1) + fundamental_frequency
    rate = fs
    kwargs = {'accurate': 0, 'min_pitch': 75, 'pulse': True}
    frame_len = int(len_sec * fs / 1000)
    frame_step = int(step_sec * fs / 1000)

    #     signal = preemphasis(signals, 0.95)

    def winfunc(num, frame_len):
        win = np.arange(frame_len)
        win = 0.54 - 0.46 * np.cos(2 * np.pi * win / (frame_len - 1))
        ret = np.ones((num, frame_len))
        return ret * win

    est_val = sig.get_F_0(signal, rate, **kwargs, time_step=0)
    f0 = np.array(est_val[1])[None, :, None]
    f0 = 1.0 / f0
    est_val0 = np.array(est_val[2]) * 16000
    shape = est_val0.shape
    if len(shape) == 2 and shape[0] > 1:
        flag = 1
        est_val0 = est_val0.astype(np.int)
        frames = np.zeros((1, 2 * len(est_val0), frame_len))
        for i in range(0, len(est_val0), 2):
            assert est_val0[i][1] - est_val0[i][0] == 560
            frames[0, i] = signal[est_val0[i][0]:est_val0[i][0] + 400]
            frames[0, i + 1] = signal[est_val0[i][1] - 400:est_val0[i][1]]
        #     frames = frames_sig(signal, frame_len, frame_step, winfunc)  # 3 dims
        fb = get_fb(num=nfb, lowfreq=300, highfreq=8000, fs=fs, nfft=nfft)
        power_spec = powspec(frames, nfft)
        energy = np.sum(power_spec, axis=-1)
        energy = np.where(energy == 0, np.finfo(float).eps, energy)
        feat = np.dot(power_spec[:, :], fb.T)
        feat = np.where(feat == 0, np.finfo(float).eps, feat)
        feat = np.log(feat)
        feat = dct(feat, type=2, axis=-1, norm='ortho')[:, :, :numcep]
        feat = lifter(feat, ceplifter)
        if appendEnergy:
            feat[:, :, 0] = np.log(energy)
        if delta > 0:
            delta1 = deltas(feat[0])[None, :]
            feat = np.concatenate((feat, delta1), axis=-1)
        if delta > 1:
            delta2 = deltas(delta1[0])[None, :]
            feat = np.concatenate((feat, delta2), axis=-1)
        if fundamental_frequency:
            f0 = np.repeat(f0, 2, axis=-2)
            feat = np.concatenate((feat, f0), axis=-1)

    else:
        flag = 0
        feat = np.zeros((1, 2, feature_length))

    return feat, flag



def get_mfcc_f0(signals,flags, length, fp, fs=16000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav_file_path = fp
    orig_wav_files = sorted(os.listdir(wav_file_path))
    for orig_wav_file in tqdm(orig_wav_files):
        if orig_wav_file[-3:] in ['wav','txt']:
            continue
        for wav in sorted(os.listdir(os.path.join(wav_file_path,orig_wav_file))):
            if wav[-3:]!='wav':
                pass
            else:
                signal, sr = librosa.load(os.path.join(wav_file_path,orig_wav_file,wav), fs)
                feature,flag = mfcc_f0(signal)
                feature = feature[0]
                flags.append(flag)
                signals.append( feature )
                length.append( len(feature) )
                # signals.append( mfcc_deltas(signal) )
    return

def get_signal_final(signals,flags, length, fp, fs=16000,delta=0, fundamental_frequency=False, m=13, ceplifter=22,flags_label=False):
    wav_file_path = fp
    orig_wav_files = sorted(os.listdir(wav_file_path))
    num  = 0
    for orig_wav_file in tqdm(orig_wav_files):
        if orig_wav_file[-3:] in ['wav','txt']:
            continue
        for wav in sorted(os.listdir(os.path.join(wav_file_path,orig_wav_file))):
            if wav[-3:]!='wav':
                pass
            else:
                if not flags_label or flags_label[num]==0:
                    feature_length = m*(1+delta) + fundamental_frequency
                    feature = np.zeros((2,feature_length))
                    flags.append(0)
                    signals.append(feature)
                    length.append(len(feature))
                else:
                    signal, sr = librosa.load(os.path.join(wav_file_path,orig_wav_file,wav), fs)
                    feature,flag = feature_final(signal, delta=delta, fundamental_frequency=fundamental_frequency, m=m, ceplifter=ceplifter)
                    feature = feature[0]
                    flags.append(flag)
                    signals.append( feature )
                    length.append( len(feature) )
                    # signals.append( mfcc_deltas(signal) )

                num+=1
    return

def feature_functionals(features, fp, csv_path=r"C:\Users\DELL\Desktop\functionals.csv", smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
), fs=16000):
    wav_file_path = fp
    orig_wav_files = sorted(os.listdir(wav_file_path))
    for orig_wav_file in tqdm(orig_wav_files):
        if orig_wav_file[-3:] in ['wav','txt']:
            continue
        for wav in sorted(os.listdir(os.path.join(wav_file_path,orig_wav_file))):
            if wav[-3:]!='wav':
                pass
            else:
                signal, sampling_rate = audiofile.read(
                    os.path.join(wav_file_path,orig_wav_file,wav),
                    duration=10,
                    always_2d=True,
                )
                feature = smile.process_signal(
                    signal,
                    sampling_rate
                )
                feature = feature.set_index([pd.Index([wav])])
                features = pd.concat([features,feature])
    return features

def get_mfcc_f0_delta(signals,flags, length, fp, fs=16000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav_file_path = fp
    orig_wav_files = sorted(os.listdir(wav_file_path))
    for orig_wav_file in tqdm(orig_wav_files):
        if orig_wav_file[-3:] in ['wav','txt']:
            continue
        for wav in sorted(os.listdir(os.path.join(wav_file_path,orig_wav_file))):
            if wav[-3:]!='wav':
                pass
            else:
                signal, sr = librosa.load(os.path.join(wav_file_path,orig_wav_file,wav), fs)
                feature,flag = mfcc_f0(signal)
                feature = feature[0]
                delta1 = deltas(feature)
                feature = np.concatenate((feature,delta1),axis=-1)
                flags.append(flag)
                signals.append( feature )
                length.append( len(feature) )
                # signals.append( mfcc_deltas(signal) )
    return

def get_another_mfccDelta(signals,flags, length, fp, fs=16000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav_file_path = fp
    orig_wav_files = sorted(os.listdir(wav_file_path))
    for orig_wav_file in tqdm(orig_wav_files):
        if orig_wav_file[-3:] in ['wav','txt']:
            continue
        for wav in sorted(os.listdir(os.path.join(wav_file_path,orig_wav_file))):
            if wav[-3:]!='wav':
                pass
            else:
                signal, sr = librosa.load(os.path.join(wav_file_path,orig_wav_file,wav), fs)
                feature,flag = another_mfcc_delta(signal)
                feature = feature[0]
                delta1 = deltas(feature)
                feature = np.concatenate((feature,delta1),axis=-1)
                flags.append(flag)
                signals.append( feature )
                length.append( len(feature) )
                # signals.append( mfcc_deltas(signal) )
    return


# append signal to signals
def get_signal(signals,fp, fs=16000,delta=2 ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav_file_path = fp
    orig_wav_files = sorted(os.listdir(wav_file_path))
    for orig_wav_file in tqdm(orig_wav_files):
        if orig_wav_file[-3:] in ['wav','txt']:
            continue
        for wav in sorted(os.listdir(os.path.join(wav_file_path,orig_wav_file))):
            if wav[-3:]!='wav':
                pass
            else:
                signal, sr = librosa.load(os.path.join(wav_file_path,orig_wav_file,wav), fs)
                signals.append( torch.tensor( mfcc_deltas(signal,delta) ).type(torch.float32) )
                # signals.append( mfcc_deltas(signal) )
    return

def get_mfcc(signals,fp, fs=16000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav_file_path = fp
    orig_wav_files = sorted(os.listdir(wav_file_path))
    for orig_wav_file in tqdm(orig_wav_files):
        if orig_wav_file[-3:] in ['wav','txt']:
            continue
        for wav in sorted(os.listdir(os.path.join(wav_file_path,orig_wav_file))):
            if wav[-3:]!='wav':
                pass
            else:
                signal, sr = librosa.load(os.path.join(wav_file_path,orig_wav_file,wav), fs)
                signals.append( torch.tensor( mfcc(signal[None,])[0] ).type(torch.float32) )
                # signals.append( mfcc_deltas(signal) )
    return

def get_signal_numpy(signals,length, fp, fs=16000,delta=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav_file_path = fp
    orig_wav_files = sorted(os.listdir(wav_file_path))
    for orig_wav_file in tqdm(orig_wav_files):
        if orig_wav_file[-3:] in ['wav','txt']:
            continue
        for wav in sorted(os.listdir(os.path.join(wav_file_path,orig_wav_file))):
            if wav[-3:]!='wav':
                pass
            else:
                signal, sr = librosa.load(os.path.join(wav_file_path,orig_wav_file,wav), fs)
                feature = mfcc_deltas(signal,delta)
                signals.append( feature )
                length.append( len(feature) )
                # signals.append( mfcc_deltas(signal) )
    return

def get_another_mfcc(signals,flags,length, fp, fs=16000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav_file_path = fp
    orig_wav_files = sorted(os.listdir(wav_file_path))
    for orig_wav_file in tqdm(orig_wav_files):
        if orig_wav_file[-3:] in ['wav','txt']:
            continue
        for wav in sorted(os.listdir(os.path.join(wav_file_path,orig_wav_file))):
            if wav[-3:]!='wav':
                pass
            else:
                (sr,signal ) = scipy.io.wavfile.read(os.path.join(wav_file_path,orig_wav_file,wav))
                signal = get_voiced_part(signal)
                if len(signal) >0:
                    feature = python_speech_features.mfcc(signal,sr)
                    flag = np.ones(feature.shape[0]).tolist()
                else:
                    feature = np.zeros((1,13))
                    flag = np.zeros(feature.shape[0]).tolist()
                signals.append( feature )
                length.append( len(feature) )

                flags.append(flag)
                # signals.append( mfcc_deltas(signal) )
    return

def get_another_mfcc_delta(signals,flags,length, fp, fs=16000,n_mfcc=13):
    wav_file_path = fp
    orig_wav_files = sorted(os.listdir(wav_file_path))
    for orig_wav_file in tqdm(orig_wav_files):
        if orig_wav_file[-3:] in ['wav','txt']:
            continue
        for wav in sorted(os.listdir(os.path.join(wav_file_path,orig_wav_file))):
            if wav[-3:]!='wav':
                pass
            else:
                (sr,signal) = scipy.io.wavfile.read(os.path.join(wav_file_path,orig_wav_file,wav))
                signal = get_voiced_part(signal)
                if len(signal) >0:
                    feature = librosa.feature.mfcc(signal,sr=fs,n_mfcc=n_mfcc)
                    # delta1 = librosa.feature.delta(feature)
                    # delta2 = librosa.feature.delta(feature,order=2)
                    # feature = np.concatenate((feature,delta1),axis=0)
                    # feature = np.concatenate((feature,delta2),axis=0)
                    # feature = feature.T
                    feature = feature.T
                    delta1 = deltas(feature)
                    feature = np.concatenate((feature,delta1),axis=-1)
                    flag = np.ones(feature.shape[0]).tolist()
                else:
                    feature = np.zeros((1,39))
                    flag = np.zeros(feature.shape[0]).tolist()
                signals.append( feature )
                length.append( len(feature) )

                flags.append(flag)
                # signals.append( mfcc_deltas(signal) )
    return

def get_voiced_part(signal,rate=16000,kwargs = { 'accurate'  : 0, 'min_pitch' : 75 ,'pulse':True}):
    '''return: 0: pitch countour, 1:time interval'''
    signal = np.array(signal)
    times = np.array((sig.get_F_0( signal, rate, **kwargs )[ 2 ]))*16000
    times = times.astype(np.int32)
    time_list = []
    out_signal = np.array([])
    if len(times) > 0:
        begin,end = times[0,0],times[0,1]
        for interval in times[1:]:
            if end>=interval[0]:
                end = interval[1]
                if end == times[-1,1]:
                    out_signal = np.concatenate((out_signal, signal[begin:end]))
            else:
                out_signal = np.concatenate((out_signal,signal[begin:end]))
                begin,end = interval
    return out_signal

def get_mfcc_numpy(signals,length, fp, fs=16000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav_file_path = fp
    orig_wav_files = sorted(os.listdir(wav_file_path))
    for orig_wav_file in tqdm(orig_wav_files):
        if orig_wav_file[-3:] in ['wav','txt']:
            continue
        for wav in sorted(os.listdir(os.path.join(wav_file_path,orig_wav_file))):
            if wav[-3:]!='wav':
                pass
            else:
                signal, sr = librosa.load(os.path.join(wav_file_path,orig_wav_file,wav), fs)
                feature = mfcc(signal[None,])[0]
                signals.append( feature )
                length.append( len(feature) )
                # signals.append( mfcc_deltas(signal) )
    return

def get_mfcc_deltas_numpy(signals,length, fp, fs=16000,delta=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav_file_path = fp
    orig_wav_files = sorted(os.listdir(wav_file_path))
    for orig_wav_file in tqdm(orig_wav_files):
        if orig_wav_file[-3:] in ['wav','txt']:
            continue
        for wav in sorted(os.listdir(os.path.join(wav_file_path,orig_wav_file))):
            if wav[-3:]!='wav':
                pass
            else:
                signal, sr = librosa.load(os.path.join(wav_file_path,orig_wav_file,wav), fs)
                feature = mfcc_deltas(signal,delta)
                signals.append( feature )
                length.append( len(feature) )
                # signals.append( mfcc_deltas(signal) )
    return

def get_original_signal(signals,fp, fs=16000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav_file_path = fp
    orig_wav_files = sorted(os.listdir(wav_file_path))
    for orig_wav_file in tqdm(orig_wav_files):
        if orig_wav_file[-3:] in ['wav','txt']:
            continue
        for wav in sorted(os.listdir(os.path.join(wav_file_path,orig_wav_file))):
            if wav[-3:]!='wav':
                pass
            else:
                signal, sr = librosa.load(os.path.join(wav_file_path,orig_wav_file,wav), fs)
                signals.append( signal )
    return

from savelist import savelist
def get_text(texts,file_path):
    def modify(text):
        text = text.upper()
        for i_text in range(len(text)):
            if text[i_text] in ['?', '!', '.', '@', '#', ',' , ':']:
                text = text[:i_text] + ' ' + text[i_text + 1:]
        return text
    def extract(text,texts,eval,flag):
        if 'Ses' not in text or flag not in text:
            return
        i = 0
        while(i<len(text)):
            if(text[i] not in [ ']','[' ]):
                i+=1
                continue
            elif (text[i]=='['):
                if text[:i-1] not in eval:
                    return
                i+=1
                continue
            elif(text[i]==']'):
                texts.append(modify(text[i+3:]))
                return
    for file in os.listdir(file_path):
        if file[-3:] != 'txt':
            continue
        path = os.path.join(file_path,file)
        eval_path = path[:49]+'EmoEvaluation'+path[63:]
        with open(path, "r", encoding="utf-8") as f2:
            eval = f2.read()
        with open(path, "r", encoding="utf-8") as f:
            k_list = savelist.read(f.read())
        for flag in ['_F0','_M0']:
            for text in k_list:
                extract(text,texts,eval,flag)
    return

def get_text_then_generate_speech(texts,file_path):
    def modify(text):
        for i_text in range(len(text)):
            if text[i_text] in ['?', '!', '.', '@', '#', ',' , ':']:
                text = text[:i_text] + ' ' + text[i_text + 1:]
        return text
    def extract(text,texts,eval,flag):
        if 'Ses' not in text or flag not in text:
            return None
        i = 0
        while(i<len(text)):
            if(text[i] not in [ ']','[' ]):
                i+=1
                continue
            elif (text[i]=='['):
                if text[:i-1] not in eval:
                    return
                i+=1
                continue
            elif(text[i]==']'):
                modified_text = modify(text[i+3:])
                texts.append(modified_text)
                return modified_text
    for file in tqdm(os.listdir(file_path)):
        if file[-3:] != 'txt':
            continue
        path = os.path.join(file_path,file)
        wav_file_path = path[:-4]
#         print(path[:-])
        Path(wav_file_path).mkdir(parents=True, exist_ok=True)
        eval_path = path[:49]+'EmoEvaluation'+path[63:]
        with open(path, "r", encoding="utf-8") as f2:
            eval = f2.read()
        with open(path, "r", encoding="utf-8") as f:
            k_list = savelist.read(f.read())
        for flag in ['_F0','_M0']:
            n = 0
            gender = 0 if flag=='_M0' else 1
            for text in k_list:
                modified_text = extract(text,texts,eval,flag)
                if modified_text != None:
                    str = f'{n}'.zfill(3)
                    text2speech(modified_text,gender,wav_file_path+f'\\{file[:-4]}_{flag[1]}{str}.wav')
                    n += 1
    return

def get_text2speech(texts,file_path):
    def modify(text,length):
        text = text.upper()
        length = len(text)
        i_text=0
        while(length>0):
            if text[i_text] in ['?', '!', '.', '@', '#', ',' , ':']:
                text = text[:i_text] + ' ' + text[i_text + 1:]
            if text[i_text]=='\'':
                text = text[:i_text]  + text[i_text + 1:]
                length -= 1
                i_text-=1
            length -= 1
            i_text+=1
        return text
    def extract(text,texts,eval,flag):
        if 'Ses' not in text or flag not in text:
            return
        i = 0
        length = len(text)
        while(i<length):
            if(text[i] not in [ ']','[' ]):
                i+=1
                continue
            elif (text[i]=='['):
                if text[:i-1] not in eval:
                    return
                i+=1
                continue
            elif(text[i]==']'):
                texts.append(modify(text[i+3:]))
                return
    for file in os.listdir(file_path):
        path = os.path.join(file_path,file)
        eval_path = path[:49]+'EmoEvaluation'+path[63:]
        with open(path, "r", encoding="utf-8") as f2:
            eval = f2.read()
        with open(path, "r", encoding="utf-8") as f:
            k_list = savelist.read(f.read())
        for flag in ['_F0','_M0']:
            for text in k_list:
                extract(text,texts,eval,flag)
    return

def get_label_4cat(labels, flags,file_path):
    if 'Session1' in file_path:
        a,b,c,d = 0.6484375 , 0.41048035, 0.30412371, 0.
    elif 'Session2' in file_path:
        a,b,c,d = 0.67679558, 0.1459854 , 0.40609137, 0.
    elif 'Session3' in file_path:
        a,b,c,d = 0.578125  , 0.4375    , 0.55737705, 0.
    elif 'Session4' in file_path:
        a,b,c,d = 0.74806202, 0.80122324, 0.54545455, 0.
    elif 'Session5' in file_path:
        a,b,c,d = 0.62760417, 0.15882353, 0.41632653, 0.

    a,b,c,d=-1,-1,-1,-1

    def extract(line,labels,flags,i):
        if 'Ses' in line:
            label = line[-28:-25]
            arr = np.zeros((4))
            if label in ['neu','ang','sad','hap']:
                # labels.append(label)
                #arr = np.ones((4))*0.225
                np.random.seed(i+1)
                rd = np.random.rand()
                if label =='neu':
                    if rd<a:
                        flags.append(0)
                        # arr[0] += 1
                        labels.append(arr)
                        return
                    arr[0] = 1
                elif label=='ang':
                    if rd<b:
                        flags.append(0)
                        # arr[1] += 1
                        labels.append(arr)
                        return
                    arr[1] += 1
                elif label=='sad':
                    if rd<c:
                        flags.append(0)
                        # arr[2] += 1
                        labels.append(arr)
                        return
                    arr[2] += 1
                elif label=='hap':
                    if rd<d:
                        flags.append(0)
                        # arr[3] += 1
                        labels.append(arr)
                        return
                    arr[3] += 1
                flags.append(1)
                labels.append(arr)
            else:
                flags.append(0)
                labels.append(arr)
    for file in os.listdir(file_path):
        if file[-3:]!='txt':
            continue
        path = os.path.join(file_path,file)
        with open(path, "r", encoding="utf-8") as f:
            k_list = savelist.read(f.read())
        for i in range(len(k_list)):
            extract(k_list[i],labels,flags,i)
    return

def get_label_4cat_ML(labels, flags,file_path):
    a,b,c,d = 0,0,0,0
    def extract(line,labels,flags,i):
        if 'Ses' in line:
            label = line[-28:-25]
            arr = 0
            if label in ['neu','ang','sad','hap']:
                # labels.append(label)
                #arr = np.ones((4))*0.225
                np.random.seed(i+1)
                # rd = np.random.rand()
                rd = 1
                if label =='neu':
                    if rd<a:
                        flags.append(0)
                        # arr[0] += 1
                        labels.append(arr)
                        return
                    arr = 0
                elif label=='ang':
                    if rd<b:
                        flags.append(0)
                        # arr[1] += 1
                        labels.append(arr)
                        return
                    arr = 1
                elif label=='sad':
                    if rd<c:
                        flags.append(0)
                        # arr[2] += 1
                        labels.append(arr)
                        return
                    arr = 2
                elif label=='hap':
                    if rd<d:
                        flags.append(0)
                        # arr[3] += 1
                        labels.append(arr)
                        return
                    arr = 4
                flags.append(1)
                labels.append(arr)
            else:
                flags.append(0)
                labels.append(arr)
    for file in os.listdir(file_path):
        if file[-3:]!='txt':
            continue
        path = os.path.join(file_path,file)
        with open(path, "r", encoding="utf-8") as f:
            k_list = savelist.read(f.read())
        for i in range(len(k_list)):
            extract(k_list[i],labels,flags,i)
    return



def get_label_ML(labels, flags,file_path):
    a,b,c,d = 0,0,0,0
    dict = {'neu': 1, 'sad': 2, 'fea': 3, 'fru': 4, 'ang': 5, 'sur': 6, 'dis': 7, 'hap': 8, 'exc': 9}
    def extract(line,labels,flags,i):
        if 'Ses' in line:
            label = line[-28:-25]
            arr = 0
            if label in ['neu','sad','fea','fru','ang','sur','dis','hap','exc']:
                arr = dict[label]
                flags.append(1)
                labels.append(arr)
            else:
                flags.append(0)
                labels.append(arr)
    for file in os.listdir(file_path):
        if file[-3:]!='txt':
            continue
        path = os.path.join(file_path,file)
        with open(path, "r", encoding="utf-8") as f:
            k_list = savelist.read(f.read())
        for i in range(len(k_list)):
            extract(k_list[i],labels,flags,i)
    return


from transformers import Wav2Vec2Processor
def get_processed_signal(signals,fp, fs=16000):
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav_file_path = fp
    orig_wav_files = sorted(os.listdir(wav_file_path))
    for orig_wav_file in tqdm(orig_wav_files):
        for wav in sorted(os.listdir(os.path.join(wav_file_path,orig_wav_file))):
            if wav[-3:]!='wav':
                pass
            else:
                signal, sr = librosa.load(os.path.join(wav_file_path,orig_wav_file,wav), fs)
                y = processor(signal, sampling_rate=fs)
                y = y['input_values'][0]
                # run through model
                # with torch.no_grad():
                y = torch.from_numpy(y)[0].to(device)
                signals.append( y )
    return


def get_classify_signal(s):
    texts=[]
    get_text(texts,fr"D:\pycharmProject\FYP\IEMOCAP语料库\Session{s}\dialog\transcriptions")
    label_cat,flags=[],[]
    get_label_4cat(label_cat,flags,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{s}\\dialog\\EmoEvaluation')
    signals = []
    get_original_signal(signals, f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{s}\\sentences\\wav')
    imitate_signals = []
    get_original_signal(imitate_signals, fr"D:\pycharmProject\FYP\IEMOCAP语料库\Session{s}\dialog\imitate")

    flags = np.array(flags)
    signals = np.array(signals)
    # label_cat = np.array(label_cat)
    idx = flags==1
    signals = signals[idx]

    imitate_signals = np.array(imitate_signals)
    imitate_signals = imitate_signals[idx]

    texts = np.array(texts)
    texts = texts[idx]

    label_cat = np.array(label_cat)
    label_cat = label_cat[idx]
    # signals = signals.tolist()
    labels = np.argmax(label_cat,axis=1)
    neural = labels==0
    angry = labels==1
    sad = labels==2
    happy = labels==3
    neu_s = signals[neural]
    ang_s = signals[angry]
    sad_s = signals[sad]
    hap_s = signals[happy]
    neu_im = imitate_signals[neural]
    ang_im = imitate_signals[angry]
    sad_im = imitate_signals[sad]
    hap_im = imitate_signals[happy]
    return signals,imitate_signals, labels,texts,neu_s,ang_s,sad_s,hap_s,neu_im,ang_im,sad_im,hap_im


def opensmile_mfcc(labels, flags,file_path):
    dict = {'neu': 1, 'sad': 2, 'fea': 3, 'fru': 4, 'ang': 5, 'sur': 6, 'dis': 7, 'hap': 8, 'exc': 9,'xxx': 0}
    def extract(line,labels,flags,i):
        if 'Ses' in line:
            label = line[-28:-25]
            arr = 0
            th = line[line.rfind('S'):-29]
            pa = file_path[: file_path.rfind('EmoEvaluation')]
            end_path = pa+th
            if label in ['neu','sad','fea','fru','ang','sur','dis','hap','exc','xxx']:
                arr = dict[label]
                flags.append(1)
                labels.append(arr)
            else:
                flags.append(0)
                labels.append(arr)
    for file in os.listdir(file_path):
        if file[-3:]!='txt':
            continue
        path = os.path.join(file_path,file)
        with open(path, "r", encoding="utf-8") as f:
            k_list = savelist.read(f.read())
        for i in range(len(k_list)):
            extract(k_list[i],labels,flags,i)
    return

def librosa_mfcc(labels, flags,file_path,mfccs,n_mfcc=25,fs=16000):
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

            assert fs == sr
            feature = np.transpose(librosa.feature.mfcc(y=signal, sr=sr,n_mfcc=n_mfcc))
            delta = librosa.feature.delta(feature,order=1)
            delta_delta = librosa.feature.delta(feature,order=2)
            fd = np.concatenate([feature, delta], axis=1)
            fdd = np.concatenate([fd, delta_delta], axis=1)

            # f0, voiced_flag, voiced_prob = librosa.pyin(signal, fmin=65, fmax=2093, sr=sr)
            # fdd = librosa.pyin(signal,fmin=65,fmax=2093,sr=sr)

            # fdd = np.concatenate([fdd,f0[:,None]],axis=1)

            # assert fdd.shape[1] == 3*n_mfcc
            mfcc = pd.DataFrame(fdd)
            if label in ['neu','sad','fea','fru','ang','sur','dis','hap','exc','xxx']:
                arr = dict[label]
            else:
                return mfccs
            mfcc['label'] = np.ones((len(mfcc),1),dtype=int)*arr
            mfccs = pd.concat([mfccs,mfcc])
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


def encoder_feature(encoder, labels=[], flags=[],mfccs=pd.DataFrame(),session=1,fs=16000):
    file_path = f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{session}\\dialog\\EmoEvaluation'
    # dict = {'neu': 1, 'sad': 2, 'fea': 3, 'fru': 4, 'ang': 5, 'sur': 6, 'dis': 7, 'hap': 8, 'exc': 9,'xxx': 0}
    dict = {'neu':1,'sad':2 , 'hap':3,'fru':4,'ang':5,'exc':6}
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

            assert fs == sr
            signal = torch.tensor(signal,dtype=torch.float32).unsqueeze(0)
            mfcc = encoder(signal).T
            mfcc = mfcc.detach().numpy()
            mfcc = pd.DataFrame(mfcc)
            if label in dict.keys():
                arr = dict[label]
            else:
                return mfccs
            mfcc['label'] = np.ones((len(mfcc),1),dtype=int)*arr
            mfcc.insert(0, 'duration', np.arange(len(mfcc)))
            mfccs = pd.concat([mfccs,mfcc])
            if len(mfccs)>1000000:
                return mfccs
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

def load_model(folder):
    models = {}
    model_path = None
    min_loss = 1000000.
    max_acc = 0
    for file in os.listdir(folder):
        if '.pth' not in file:
            continue
        id = file.index('loss')
        loss = float(file[id + 5:id + 10])
        acc = int(file[id-4:id-1])
        if acc>max_acc:
            max_acc = acc
        models[os.path.join(folder,file)] = loss
    model_list = [k for k, v in sorted(models.items(), key=lambda item: item[1],reverse=True)]
    if len(model_list)>0:
        model_path = model_list[-1]
        # if loss<min_loss:
        #     min_loss = loss
        #     model_path = os.path.join(folder,file)
    print(model_path)
    if model_path is not None:
        model = torch.load(model_path)
        print('find available')
    else:
        model = None
        print('no available')
    print(f'Model={model_path}')
    return model,min_loss,max_acc,model_list

def load_best_model(folder):
    model_path = None
    max_acc = 0
    for file in os.listdir(folder):
        if '.pth' not in file:
            continue
        id = file.index('loss')
        acc = int(file[id-4:id-1])
        if acc>max_acc:
            max_acc = acc
            model_path = os.path.join(folder,file)
    model = torch.load(model_path)
    return model

def raw_audio(labels=[], flags=[],mfccs=pd.DataFrame(),session=1,fs=16000):
    file_path = f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{session}\\dialog\\EmoEvaluation'
    # dict = {'neu': 1, 'sad': 2, 'fea': 3, 'fru': 4, 'ang': 5, 'sur': 6, 'dis': 7, 'hap': 8, 'exc': 9,'xxx': 0}
    dict = {'neu':1,'sad':2 , 'hap':3,'fru':4,'ang':5,'exc':6}
    def extract(line,labels,flags,i,path,mfccs,lengths):
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
            lengths.append(len(signal))
            pad = np.zeros(466080-len(signal),dtype=np.float32)
            signal = np.concatenate([signal,pad],dtype=np.float32)

            assert fs == sr
            mfcc = signal[None,:]
            mfcc = pd.DataFrame(mfcc,dtype=np.float32)
            if label in dict.keys():
                arr = dict[label]
            else:
                return mfccs
            mfcc['label'] = np.array(arr)
            mfcc.insert(0, 'duration', lengths[-1])
            mfccs = pd.concat([mfccs,mfcc],axis=0)
            # print(mfccs.shape)
            if len(mfccs)>1000000:
                return mfccs
            return mfccs

    lengths=[]
    for file in tqdm(os.listdir(file_path)):
        if file[-3:]!='txt':
            continue
        path = os.path.join(file_path,file)
        with open(path, "r", encoding="utf-8") as f:
            k_list = savelist.read(f.read())
        for i in range(len(k_list)):
            mfccs = extract(k_list[i],labels,flags,i,path,mfccs,lengths=lengths)
    print(max(lengths))
    return mfccs

def get_raw(labels=[],labels_test=[], flags=[],mfccs=[],mfccs_test=[],session=1,fs=16000,test_size=0.2):
    file_path = f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{session}\\dialog\\EmoEvaluation'
    # dict = {'neu': 1, 'sad': 2, 'fea': 3, 'fru': 4, 'ang': 5, 'sur': 6, 'dis': 7, 'hap': 8, 'exc': 9,'xxx': 0}
    dict = {'neu':1,'sad':2 , 'hap':3,'fru':4,'ang':5,'exc':6}
    def extract(line,labels,labels_test,flags,i,path,mfccs,mfccs_test,lengths):
        if 'Ses' not in line:
            return mfccs,mfccs_test,labels,labels_test
        else:
            label = line[-28:-25]
            th = line[line.rfind('S'):-29]
            pa = file_path[: file_path.rfind('dialog')]
            name = path[path.rfind('Ses'):path.rfind('.txt')]
            end_path = pa+'sentences\\wav\\'+name+'\\'+th+'.wav' #"D:\pycharmProject\FYP\IEMOCAP语料库\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F000.wav"
            signal, sr = librosa.load(end_path,sr=fs)
            if label in dict.keys():
                arr = dict[label]
            else:
                return mfccs,mfccs_test,labels,labels_test
            cnt = len(mfccs) + len(mfccs_test)
            np.random.seed(cnt)
            if np.random.rand() > test_size:
                mfccs.append(signal)
                labels.append(arr)
            else:
                mfccs_test.append(signal)
                labels_test.append(arr)

            # mfccs.append(signal)
            # if len(mfccs)>1000000:
            #     return mfccs
            return mfccs,mfccs_test,labels,labels_test

    lengths=[]
    for file in tqdm(os.listdir(file_path)):
        if file[-3:]!='txt':
            continue
        path = os.path.join(file_path,file)
        with open(path, "r", encoding="utf-8") as f:
            k_list = savelist.read(f.read())
        for i in range(len(k_list)):
            mfccs,mfccs_test,labels,labels_test = extract(k_list[i],labels,labels_test,flags,i,path,mfccs,mfccs_test,lengths=lengths)
    return mfccs,mfccs_test,torch.tensor(labels).to('cuda')-1,torch.tensor(labels_test).to('cuda')-1

def get_raw_SAVEE(labels=[],labels_test=[],mfccs=[],mfccs_test=[],fs=16000,test_size=0.2):
    root_path = r"E:\SAVEE\AudioData"
    dict = {'n': 1, 'f': 2, 'a': 3, 'd': 4,
            'h': 5}  # num2label3 = ['neural','fear','anger','disgust','happiness','sad','surprise']
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
            cnt = len(mfccs) + len(mfccs_test)
            np.random.seed(cnt)
            if np.random.rand() > test_size:
                mfccs.append(signal)
                labels.append(arr)
            else:
                mfccs_test.append(signal)
                labels_test.append(arr)
    return mfccs,mfccs_test,torch.tensor(labels).to('cuda')-1,torch.tensor(labels_test).to('cuda')-1

def voiced_index(df):
    time = np.array(df.iloc[:,0])
    starts = time== 0
    starts = np.arange(len(starts))[starts]
    total_voiced = np.array([])
    index = np.array([])
    for i in range(len(starts)):
        if i == len(starts)-1:
            ed = len(time)
        else:
            ed = starts[i+1]
        bg = starts[i]
        voiced_part = np.array(df['flag'][bg:ed])
        total_voiced = np.concatenate([total_voiced,voiced_part])
        index = np.concatenate([index,np.arange((voiced_part==True).sum())])
        voiced_part[1:] += voiced_part[:-1]
        voiced_part[1:] += voiced_part[:-1]
        voiced_part[1:] += voiced_part[:-1]
    # print(np.unique(total_voiced!=0,return_counts=True))
    total_voiced = np.arange(len(total_voiced))[total_voiced!=0]
    return total_voiced, index


def get_raw_transformer(labels=[],labels_test=[], flags=[],mfccs=[],mfccs_test=[],session=1,fs=16000,test_size=0.2):
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    file_path = f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{session}\\dialog\\EmoEvaluation'
    # dict = {'neu': 1, 'sad': 2, 'fea': 3, 'fru': 4, 'ang': 5, 'sur': 6, 'dis': 7, 'hap': 8, 'exc': 9,'xxx': 0}
    dict = {'neu':1,'sad':2 , 'hap':3,'fru':4,'ang':5,'exc':6}
    def extract(line,labels,labels_test,flags,i,path,mfccs,mfccs_test,lengths):
        if 'Ses' not in line:
            return mfccs,mfccs_test,labels,labels_test
        else:
            label = line[-28:-25]
            th = line[line.rfind('S'):-29]
            pa = file_path[: file_path.rfind('dialog')]
            name = path[path.rfind('Ses'):path.rfind('.txt')]
            end_path = pa+'sentences\\wav\\'+name+'\\'+th+'.wav' #"D:\pycharmProject\FYP\IEMOCAP语料库\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F000.wav"
            signal, sr = librosa.load(end_path,sr=fs)
            y = processor(signal, sampling_rate=sr)
            y = y['input_values'][0]
            if len(y)==1:
                signal = y[0]
            else:
                signal = y
            if label in dict.keys():
                arr = dict[label]
            else:
                return mfccs,mfccs_test,labels,labels_test
            cnt = len(mfccs) + len(mfccs_test)
            np.random.seed(cnt)
            if np.random.rand() > test_size:
                mfccs.append(signal)
                labels.append(arr)
            else:
                mfccs_test.append(signal)
                labels_test.append(arr)

            # mfccs.append(signal)
            # if len(mfccs)>1000000:
            #     return mfccs
            return mfccs,mfccs_test,labels,labels_test

    lengths=[]
    for file in tqdm(os.listdir(file_path)):
        if file[-3:]!='txt':
            continue
        path = os.path.join(file_path,file)
        with open(path, "r", encoding="utf-8") as f:
            k_list = savelist.read(f.read())
        for i in range(len(k_list)):
            mfccs,mfccs_test,labels,labels_test = extract(k_list[i],labels,labels_test,flags,i,path,mfccs,mfccs_test,lengths=lengths)
    return mfccs,mfccs_test,torch.tensor(labels).to('cuda')-1,torch.tensor(labels_test).to('cuda')-1

def store_raw_transformer(labels=[],labels_test=[], flags=[],mfccs=pd.DataFrame([]),mfccs_test=[],session=1,fs=16000,test_size=0.2):
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    file_path = f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{session}\\dialog\\EmoEvaluation'
    # dict = {'neu': 1, 'sad': 2, 'fea': 3, 'fru': 4, 'ang': 5, 'sur': 6, 'dis': 7, 'hap': 8, 'exc': 9,'xxx': 0}
    dict = {'neu':1,'sad':2 , 'hap':3,'fru':4,'ang':5,'exc':6}
    def extract(line,labels,labels_test,flags,i,path,mfccs,mfccs_test,lengths):
        if 'Ses' not in line:
            return mfccs,mfccs_test,labels,labels_test
        else:
            label = line[-28:-25]
            th = line[line.rfind('S'):-29]
            pa = file_path[: file_path.rfind('dialog')]
            name = path[path.rfind('Ses'):path.rfind('.txt')]
            end_path = pa+'sentences\\wav\\'+name+'\\'+th+'.wav' #"D:\pycharmProject\FYP\IEMOCAP语料库\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F000.wav"
            signal, sr = librosa.load(end_path,sr=fs)
            y = processor(signal, sampling_rate=sr)
            y = y['input_values'][0]
            if len(y)==1:
                signal = y[0]
            else:
                signal = y
            if label in dict.keys():
                arr = dict[label]
            else:
                return mfccs,mfccs_test,labels,labels_test
            dim = np.array([float(line[-15:-9]),float(line[-7:-1]),float(line[-23:-17])],dtype=np.float32)
            signal = np.concatenate([dim,signal],dtype=np.float32)
            mfcc = pd.DataFrame({arr-1:signal})
            mfccs = pd.concat([mfccs,mfcc],axis=1)
            # print(mfccs)

            # mfccs.append(signal)
            # if len(mfccs)>1000000:
            #     return mfccs
            return mfccs,mfccs_test,labels,labels_test
    lengths=[]
    for file in tqdm(os.listdir(file_path)):
        if file[-3:]!='txt':
            continue
        path = os.path.join(file_path,file)
        with open(path, "r", encoding="utf-8") as f:
            k_list = savelist.read(f.read())
        for i in range(len(k_list)):
            mfccs,mfccs_test,labels,labels_test = extract(k_list[i],labels,labels_test,flags,i,path,mfccs,mfccs_test,lengths=lengths)
    mfccs.to_csv(r"E:\FYP excel\IEMOCAP1_Transformer_data.csv")
    return mfccs,torch.tensor(labels).to('cuda')-1


def store_raw_SAVEE_transformer(labels=[],labels_test=[],mfccs=pd.DataFrame([]),mfccs_test=[],fs=16000,test_size=0.2):
    root_path = r"E:\SAVEE\AudioData"
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    dict = {'n': 1, 'f': 2, 'a': 3, 'd': 4,
            'h': 5}  # num2label3 = ['neural','fear','anger','disgust','happiness','sad','surprise']
    for file in os.listdir(root_path):
        if '.' in file:
            continue
        person_path = os.path.join(root_path, file)
        for wav_path in tqdm(os.listdir(person_path)):
            wav = os.path.join(person_path, wav_path)

            signal, sr = librosa.load(wav, sr=fs)
            y = processor(signal, sampling_rate=sr)
            y = y['input_values'][0]
            if len(y) == 1:
                signal = y[0]
            else:
                signal = y
            if wav_path[0] == 's':
                if wav_path[1] == 'a':
                    arr = 6
                elif wav_path[1] == 'u':
                    arr = 7
            else:
                arr = dict[wav_path[0]]

            assert fs == sr
            mfcc = pd.DataFrame({arr-1:signal})
            mfccs = pd.concat([mfccs,mfcc],axis=1)
    mfccs.to_csv(r"E:\FYP excel\SAVEE_Transformer_data.csv")
    return mfccs,mfccs_test,torch.tensor(labels).to('cuda')-1,torch.tensor(labels_test).to('cuda')-1

def get_raw_SAVEE_transformer(labels=[],labels_test=[],mfccs=[],mfccs_test=[],fs=16000,test_size=0.2):
    root_path = r"E:\SAVEE\AudioData"
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    dict = {'n': 1, 'f': 2, 'a': 3, 'd': 4,
            'h': 5}  # num2label3 = ['neural','fear','anger','disgust','happiness','sad','surprise']
    for file in os.listdir(root_path):
        if '.' in file:
            continue
        person_path = os.path.join(root_path, file)
        for wav_path in tqdm(os.listdir(person_path)):
            wav = os.path.join(person_path, wav_path)

            signal, sr = librosa.load(wav, sr=fs)
            y = processor(signal, sampling_rate=sr)
            y = y['input_values'][0]
            if len(y) == 1:
                signal = y[0]
            else:
                signal = y
            if wav_path[0] == 's':
                if wav_path[1] == 'a':
                    arr = 6
                elif wav_path[1] == 'u':
                    arr = 7
            else:
                arr = dict[wav_path[0]]

            assert fs == sr
            cnt = len(mfccs) + len(mfccs_test)
            np.random.seed(cnt)
            if np.random.rand() > test_size:
                mfccs.append(signal)
                labels.append(arr)
            else:
                mfccs_test.append(signal)
                labels_test.append(arr)
    return mfccs,mfccs_test,torch.tensor(labels).to('cuda')-1,torch.tensor(labels_test).to('cuda')-1

def get_dim_IEMOCAP(labels=[],labels_test=[], flags=[],mfccs=pd.DataFrame([]),mfccs_test=[],session=1,fs=16000,test_size=0.2):
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    file_path = f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{session}\\dialog\\EmoEvaluation'
    # dict = {'neu': 1, 'sad': 2, 'fea': 3, 'fru': 4, 'ang': 5, 'sur': 6, 'dis': 7, 'hap': 8, 'exc': 9,'xxx': 0}
    dict = {'neu':1,'sad':2 , 'hap':3,'fru':4,'ang':5,'exc':6}
    def extract(line,labels,labels_test,flags,i,path,mfccs,mfccs_test,lengths):
        if 'Ses' not in line:
            return mfccs,mfccs_test,labels,labels_test
        else:
            label = line[-28:-25]
            th = line[line.rfind('S'):-29]
            pa = file_path[: file_path.rfind('dialog')]
            name = path[path.rfind('Ses'):path.rfind('.txt')]
            end_path = pa+'sentences\\wav\\'+name+'\\'+th+'.wav' #"D:\pycharmProject\FYP\IEMOCAP语料库\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F000.wav"
            dim = line[-1]
            mfcc = pd.DataFrame(dim)
            mfccs = pd.concat([mfccs,mfcc],axis=0)

            # mfccs.append(signal)
            # if len(mfccs)>1000000:
            #     return mfccs
            return mfccs,mfccs_test,labels,labels_test
    lengths=[]
    for file in tqdm(os.listdir(file_path)):
        if file[-3:]!='txt':
            continue
        path = os.path.join(file_path,file)
        with open(path, "r", encoding="utf-8") as f:
            k_list = savelist.read(f.read())
        for i in range(len(k_list)):
            mfccs,mfccs_test,labels,labels_test = extract(k_list[i],labels,labels_test,flags,i,path,mfccs,mfccs_test,lengths=lengths)
    mfccs.to_csv(r"E:\FYP excel\IEMOCAP1_Transformer_data.csv")
    return mfccs,torch.tensor(labels).to('cuda')-1


if __name__=='__main__':
    store_raw_transformer()
    # store_raw_SAVEE_transformer()
    pass