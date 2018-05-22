# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, torch as t,os
import torch as t,functools
from torch.utils import data
import os, pandas as pd
import scipy.io as sio
import numpy as np
from torchvision  import transforms
from sklearn.model_selection import  train_test_split
from PIL import Image

def read_mat(filename):
    data = sio.loadmat(filename)
    data = data['data']
    return data.astype(np.float)


class EEGDataset(data.Dataset):
    def __init__(self, folder_name, train=True, transforms=None):
        np.random.seed(1)
        '''
        Attributes:

        '''
        self.transform = transforms
        self.root = folder_name
        filenames =[x for x in os.listdir(folder_name) if x.find('.mat')>=0]
        filenames.sort()
        if train ==True:
            filenames = [x for x in filenames if not x.find('test')>0]
            targets = [0 if x.find('inter')>0 else 1 for x in filenames]
            self.filenames,_,self.targets,_=train_test_split(filenames,targets,test_size=0.35,random_state=1)
        else:
            filenames = [x for x in filenames if not x.find('test')>0]
            targets = [0 if x.find('inter')>0 else 1 for x in filenames]
            _,self.filenames,_,self.targets=train_test_split(filenames,targets,test_size=0.35,random_state=1)


    def __getitem__(self, index):
        '''
           - index: 下标，图像的序号，可以通过ix2id[index]获取对应图片文件名
        '''
        data = read_mat(os.path.join(self.root,self.filenames[index]))
        # data = Image.fromarray(data, 'RGB')
        # if self.transform is not None:
        #     data = self.transform(data)
        target = self.targets[index]
        return data, target


    def __len__(self):
        return len(self.filenames)



def get_dataloader(folder, batch_size=128, num_workers=4,training=True):
    data_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = EEGDataset(folder,train=training,transforms=data_transform)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=training,
                                 num_workers=num_workers,
                                 )

    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader('/home/jizong/Desktop/Seizure Prediction/processed_data/'
                                'fft_meanlog_std_lowcut0.1highcut180nfreq_bands6win_length_sec60stride_sec60/Dog_1')

    for ii, (data,target) in enumerate(dataloader):
        print(ii, data.shape,target.shape)