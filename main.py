# -*- coding: utf-8 -*-
from models.oneD_cnn_model import oneD_conv
from cnn_based_model.dataset_cnn import get_dataloader
from torch import nn
import torch as t,numpy as np
from tqdm import tqdm
from Visualization import Visualizer
from torchnet.meter import AverageValueMeter
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,confusion_matrix
vis = Visualizer('http://turing.livia.etsmtl.ca',env='EEG')

folder_name = 'Dog_1/features'
dataloader_train = get_dataloader(folder_name)
dataloader_test = get_dataloader(folder_name,training=False)

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
            # auc = roc_auc_score(target.cpu().numpy(),proba.cpu().numpy())
            acc = (decision==target).sum().item()/np.float(len(target))
            avg_acc.add(acc)
    avg_auc = roc_auc_score(y_true,y_predict_proba)
    net.train()
    cnf_matrix = confusion_matrix(y_true, y_predict)
    np.set_printoptions(precision=2)
    print(cnf_matrix,avg_auc)
    return avg_acc.value()[0],avg_auc

net = oneD_conv().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = t.optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-4)
epoches = 100000
# vis.reinit()
loss_avg= AverageValueMeter()
for epoch in range(epoches):
    loss_avg.reset()
    for i,(data, target) in tqdm(enumerate(dataloader_train)):
        data=data.type(t.FloatTensor)
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        loss_avg.add(loss.item())

    if epoch%5==0:
        # print(target)
        acc_train,_=val(dataloader_train,net)
        acc_test, auc_test = val(dataloader_test,net)
        vis.plot('train_acc', acc_train)
        vis.plot('test_acc', acc_test)
        vis.plot('test_auc', auc_test)
    vis.plot('loss', loss_avg.value()[0])







