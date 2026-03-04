# step of mfcc:
# a.preprocessing: 1.sampling=>frames  2.preemphasis  3.filter
# b. 1.fft  2.melbank  3.
from __future__ import division
import numpy as np
from preprocessing import *
from scipy.fftpack import dct


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
    return feat  #[ num_frame, 13 ]  随信号长度改变而变化




