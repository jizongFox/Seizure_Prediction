# -*- coding: utf-8 -*-
import torch as t, torch.nn as nn
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from cnn_based_model.dataset_cnn import get_dataloader
from cnn_based_model.model import convNet
from torchnet.meter import AverageValueMeter
from sklearn.metrics import roc_auc_score,confusion_matrix
import torch.nn.functional as F
from Visualization import Visualizer
def train():
    vis = Visualizer(server='http://turing.livia.etsmtl.ca', env='EEG')
    data_root = '/home/AN96120/python_project/Seizure Prediction/processed_data/fft_meanlog_std_lowcut0.1highcut180nfreq_bands12win_length_sec60stride_sec60/Dog_1'
    dataloader_train = get_dataloader(data_root, training=True)
    dataloader_test = get_dataloader(data_root, training=False)
    # No interaction has been found in the training and testing dataset.
    weights = t.Tensor([1/(np.array(dataloader_train.dataset.targets)==0).mean(),1/(np.array(dataloader_train.dataset.targets)==1).mean()  ])
    criterion = nn.CrossEntropyLoss(weight=weights.cuda())

    net = convNet ()
    net.cuda()

    optimiser = t.optim.Adam(net.parameters(),lr= 1e-4,weight_decay=1e-4)
    loss_avg = AverageValueMeter()
    epochs = 10000
    for epoch in range(epochs):
        loss_avg.reset()
        for ii, (data, targets) in enumerate(dataloader_train):
            data, targets= data.type(t.FloatTensor), targets.type(t.LongTensor)
            data = data.cuda()
            targets = targets.cuda()
            optimiser.zero_grad()
            output = net(data)
            loss = criterion(output,targets)
            loss_avg.add(loss.item())
            loss.backward()
            optimiser.step()
        vis.plot('loss',loss_avg.value()[0])

        _,auc_train=val(dataloader_train,net)
        _, auc_test =val(dataloader_test,net)
        print(auc_train,auc_test)

def val(dataloader, net):
    avg_acc=AverageValueMeter()
    avg_acc.reset()
    y_true =[]
    y_predict=[]
    y_predict_proba=[]
    net.eval()
    with t.no_grad():
        for i,(data,target) in enumerate(dataloader):
            data=data.type(t.FloatTensor)
            data = data.cuda()
            target = target.cuda()
            output = net(data)
            decision = output.max(1)[1]
            y_predict.extend(decision.cpu().numpy().tolist())
            proba = F.softmax(output,dim=1)[:,1]
            y_predict_proba.extend(proba.cpu().numpy().tolist())
            y_true.extend(target.cpu().numpy().tolist())
            acc = (decision==target).sum().item()/np.float(len(target))
            avg_acc.add(acc)
    avg_auc = roc_auc_score(y_true,y_predict_proba)

    cnf_matrix = confusion_matrix(y_true, y_predict)
    np.set_printoptions(precision=2)
    # print(avg_auc)
    net.train()
    return avg_acc.value()[0],avg_auc


if __name__=="__main__":
    train()
