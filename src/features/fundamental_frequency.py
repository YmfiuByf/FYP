import librosa
import IPython.display as ipd
import os
from scipy.io import matlab
import pytest
import Signal_Analysis.features.signal as sig
import numpy as np
import scipy

signal = librosa.load(r"D:\pycharmProject\FYP\IEMOCAP语料库\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F000.wav",16000)[0]

def compute_F0(signal,rate=16000,kwargs = { 'accurate'  : 0, 'min_pitch' : 75 ,'pulse':True}):
    est_val0 = np.array((sig.get_F_0( signal, rate, **kwargs )[ 0 ]))
    est_val0 = est_val0.astype(np.int32)
    print(est_val0)
    #print(np.mean(est_val0),np.var(est_val0),len(est_val0),np.max(est_val0),np.min(est_val0),np.median(est_val0))
def compute_voiced_time(signal,rate=16000,kwargs = { 'accurate'  : 0, 'min_pitch' : 75 ,'pulse':True}):
    est_val0 = np.array((sig.get_F_0( signal, rate, **kwargs )[ 2 ]))*16000
    est_val0 = est_val0.astype(np.int32)
    print(est_val0)

compute_F0(signal)