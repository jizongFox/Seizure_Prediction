# -*- coding: utf-8 -*-
import torch as t,functools
from torch.utils import data
import os, pandas as pd
from PIL import Image
import torchvision as tv
import numpy as np
from utils.utils import read_mat
from sklearn.model_selection import  train_test_split


class EEGDataset(data.Dataset):
    def __init__(self, folder_name, train=True, transforms=None):
        '''
        Attributes:
            _data (dict): 预处理之后的数据，包括所有图片的文件名，以及处理过后的描述
            all_imgs (tensor): 利用resnet50提取的图片特征，形状（200000，2048）
            caption(list): 长度为20万的list，包括每张图片的文字描述
            ix2id(dict): 指定序号的图片对应的文件名
            start_(int): 起始序号，训练集的起始序号是0，验证集的起始序号是190000，即
                前190000张图片是训练集，剩下的10000张图片是验证集
            len_(init): 数据集大小，如果是训练集，长度就是190000，验证集长度为10000
            traininig(bool): 是训练集(True),还是验证集(False)
        '''
        filenames = read_name_in_a_folder(folder_name)
        filenames.sort()
        if train ==True:
            filenames = [x for x in filenames if not x.find('test')>0]
            targets = [0 if x.find('inter')>0 else 1 for x in filenames]
            self.filenames,_,self.targets,_=train_test_split(filenames,targets,test_size=0.25,random_state=1)
        else:
            filenames = [x for x in filenames if not x.find('test')>0]
            targets = [0 if x.find('inter')>0 else 1 for x in filenames]
            _,self.filenames,_,self.targets=train_test_split(filenames,targets,test_size=0.25,random_state=1)


    def __getitem__(self, index):
        '''
        返回：
        - img: 图像features 2048的向量
        - caption: 描述，形如LongTensor([1,3,5,2]),长度取决于描述长度
        - index: 下标，图像的序号，可以通过ix2id[index]获取对应图片文件名
        '''
        # data,_ = read_mat(self.filenames[index])
        data = np.load(self.filenames[index]).T
        # data = self.transform(data)
        target = self.targets[index]
        return data, target

        # # 5句描述随机选一句
        # rdn_index = np.random.choice(len(caption), 1)[0]
        # caption = caption[rdn_index]
        # return img, t.LongTensor(caption), index

    def __len__(self):
        return len(self.filenames)

    def transform(self,data):
        sequential_len=data.shape[1]
        rnd_start = np.random.randint(0,sequential_len-400*60*8-100)
        clipped_data = data[:,rnd_start:rnd_start+400*60*8]
        return  clipped_data

def get_dataloader(folder, batch_size=32, num_workers=4,training=True):
    dataset = EEGDataset(folder,train=training)
    # weighted_sampling
    if training:
        weights =pd.Series(dataset.targets).value_counts().values/len(dataset.targets)
        weights_ = [weights[1] if x == 0 else weights[0] for x in dataset.targets]
        sampler = t.utils.data.sampler.WeightedRandomSampler(weights_, len(dataset.targets),replacement=True)

        dataloader = data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     sampler=sampler
                                     )
    else:
        dataloader = data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     )

    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader('Dog_1')


    for ii, (data,target) in enumerate(dataloader):
        print(ii, data.shape)