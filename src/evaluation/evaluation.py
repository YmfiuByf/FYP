import pandas as pd
import os
import numpy as np
from CNN import *

path = r"E:\FYP excel"
for file in os.listdir(path):
    if '_CNN_' not in file or '_k=[' not in file:
        continue
    if '.csv' in file or '.png' in file:
        continue
    if 'SAVEE' in file:
        dataset = 'SAVEE'
    elif 'IEMOCAP1' in file:
        dataset = 'IEMOCAP1'
    remove_silence = True if '_voiced_' in file else False
    file_path = os.path.join(path, file)
    max_acc = 0
    load_path=None
    for model_path in os.listdir(file_path):
        if '.pth' not in model_path:
            pass
        else:
            id = model_path.index('loss')
            acc = int(model_path[id - 4:id - 1])
            # print(f'accuracy={acc},{max_acc}')
            if acc>max_acc:
                max_acc = acc
                load_path = os.path.join(file_path,model_path)
                # print(load_path is None)
    if load_path is None:
        print('no model available',file)
        # os.remove(file_path)
        continue
    id = file.index('mfcc')
    n_mfcc = int(file[id+4:id+6])
    # print(n_mfcc)
    print('reach here')
    train_CNN(test_mode=True,n_mfcc=n_mfcc,dataset=dataset,folder=file_path,remove_silence=remove_silence)
    print('executed')


