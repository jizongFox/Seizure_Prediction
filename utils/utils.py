# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, torch as t,os
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from functools import partial
import scipy.io as sio
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=9):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def read_mat(filename):
    data = sio.loadmat(filename)
    if filename.find('inter')>0:
        data = data[list(data.keys())[3]][0][0]
        node_name= data[3][0]
        data = data[0]

    elif filename.find('preictal')>0:
        data = data[list(data.keys())[3]][0][0]
        node_name= data[3][0]
        data = data[0]

    elif filename.find('test')>0:
        data = data['test_segment_1'][0][0]
        node_name= data[3][0]
        data = data[0]
    else:
        raise AttributeError
    return data.astype(np.float), node_name

def read_name_in_a_folder(folder):
    '''
    :param folder:
    :return: all the names of .mat files with inter-, pre-, and test cases
    '''
    file_names = os.listdir(folder)
    file_names = [os.path.join(folder,x) for x in file_names if x.find('.mat')>0]
    file_names.sort()
    return file_names

def read_name_for_folders(list_of_folders):
    if type(list_of_folders)==str:
        return read_name_in_a_folder(list_of_folders)
    names=[]
    for folder in list_of_folders:
        names.extend(read_name_in_a_folder(folder))
    names.sort()
    return names


def transform(filename_fullpath):
    data,_ = read_mat(filename_fullpath)
    data=data.astype(np.float)
    for i in range(data.shape[0]):
        data[i]=(data[i]-data[i].mean())/data[i].std()

    data_= np.zeros((3,data.shape[0],data.shape[1]))

    filter_low = partial(butter_bandpass_filter,lowcut = 1, highcut=100, fs=400 )
    filter_middle = partial(butter_bandpass_filter,lowcut = 7, highcut=14, fs=400 )
    filter_high = partial(butter_bandpass_filter,lowcut = 14, highcut=49, fs=400 )
    b1=filter_low(data[1])
    b2=filter_middle(data[1])
    b3=filter_high(data[1])
    plt.plot(b1)
    plt.plot(b2)
    plt.plot(b3)
    plt.show()

    # for i in range(data.shape[0]):

    filter_low = 1

    return data_





if __name__=="__main__":
    a= read_name_for_folders(['Dog_1'])
    transform(a[0])
    pass


