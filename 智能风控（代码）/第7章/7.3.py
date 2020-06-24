# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:00:19 2019

@author: zixing.mei
"""

import torch      
import torch.nn as nn      
import random    
from sklearn.model_selection import train_test_split    
import torchvision.transforms as transforms      
import torchvision.datasets as dsets      
from torch.autograd import Variable      
      
random_st = random.choice(range(10000))      
train_images, test_images = train_test_split(train_images,test_size=0.15,   
                        random_state=random_st)      
    
train_data = MyDataset(train_images)        
test_data = MyDataset(test_images)      
    
train_loader = torch.utils.data.DataLoader(train_data, batch_size=50,   
                       shuffle=True, num_workers=0)  
test_loader = torch.utils.data.DataLoader(test_data, batch_size=25,   
                      huffle=False, num_workers=0)  
#搭建LSTM网络    
class Rnn(nn.Module):      
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):      
        super(Rnn, self).__init__()      
        self.n_layer = n_layer      
        self.hidden_dim = hidden_dim      
        self.LSTM = nn.LSTM(in_dim, hidden_dim,   
                   n_layer,batch_first=True)      
        self.linear = nn.Linear(hidden_dim,n_class)      
        self.sigmoid = nn.Sigmoid()       
       
    def forward(self, x):      
        x = x.sum(dim = 1)      
        out, _ = self.LSTM(x)      
        out = out[:, -1, :]      
        out = self.linear(out)      
        out = self.sigmoid(out)      
        return out  
#28个特征，42个月切片，2个隐层，2分类        
model = Rnn(28,42,2,2)       
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
model = model.to(device)       
#使用二分类对数损失函数      
criterion = nn.SoftMarginLoss(reduction='mean')         
opt = torch.optim.Adam(model.parameters())        
total_step = len(train_loader)        
total_step_test = len(test_loader)    
num_epochs = 50    

for epoch in range(num_epochs):      
    train_label = []      
    train_pred = []      
    model.train()      
    for i, (images, labels) in enumerate(train_loader):      
        images = images.to(device)      
        labels = labels.to(device)      
        #网络训练    
        out = model(images)      
        loss = criterion(out, labels)      
        opt.zero_grad()      
        loss.backward()      
        opt.step()      
        #每一百轮打印一次    
        if i%100 == 0:      
            print('train epoch: {}/{}, round: {}/{},loss: {}'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss))  
        #真实标记和预测值    
        train_label.extend(labels.cpu().numpy().flatten().tolist())      
        train_pred.extend(out.detach().cpu().numpy().flatten().tolist())    
    #计算真正率和假正率    
    fpr_lm_train, tpr_lm_train, _ = roc_curve(np.array(train_label),   
                                                      np.array(train_pred))      
    #计算KS和AUC     
    print('train epoch: {}/{}, KS: {}, ROC: {}'.format(      
        epoch + 1, num_epochs,abs(fpr_lm_train - tpr_lm_train).max(),  
               metrics.auc(fpr_lm_train, tpr_lm_train)))      
        
    test_label = []      
    test_pred = []      
        
    model.eval()      
    #计算测试集上的KS值和AUC值    
    for i, (images, labels) in enumerate(test_loader):      
            
        images = images.to(device)      
        labels = labels.to(device)      
        out = model(images)      
        loss = criterion(out, labels)      
            
        #计算KS和AUC      
        if i%100 == 0:      
            print('test epoch: {}/{}, round: {}/{},loss: {}'.format(
                    epoch + 1, num_epochs,i + 1, total_step_test, loss))      
        test_label.extend(labels.cpu().numpy().flatten().tolist())      
        test_pred.extend(out.detach().cpu().numpy().flatten().tolist())    
        
    fpr_lm_test, tpr_lm_test, _ = roc_curve(np.array(test_label),   
                                                    np.array(test_pred))      
        
    print('test epoch: {}/{}, KS: {}, ROC: {}'.format( epoch + 1,
                                                        num_epochs,
                                                   abs(fpr_lm_test - tpr_lm_test).max(),
                                                      metrics.auc(fpr_lm_test - tpr_lm_test)))

