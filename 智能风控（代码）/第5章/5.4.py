# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:46:56 2019

@author: zixing.mei
"""

import numpy as np  
from utils import *  
import pandas as pd  
import sklearn.svm as svm 
from collections import Counter  
class TSVM(object):  
    def __init__(self):  
         # 分别对应有label的样本权重和无label的样本权重
        self.Cu = 0.001   
        self.Cl = 1  
    def fit(self,train_data):  
        # 将数据集中的第一个正例，和第一个负例作为真实标记样本，其余视为无标记。  
        pos_one = train_data[train_data[:,0] == 1][0]  
        pos_other = train_data[train_data[:,0] == 1][1:]  
        neg_one = train_data[train_data[:,0] == -1][0]  
        neg_other = train_data[train_data[:,0] == -1][1:]  
        train = np.vstack((pos_one,neg_one))   
         #S用于对数据进行测试
        self.other = np.vstack((pos_other,neg_other))   
        # 训练一个初始的分类器，设置不均衡参数  
        self.clf =  svm.SVC(C=1.5, kernel=self.kernel)
        self.clf.fit(train[:,1:],train[:,0])  
        pred_y = self.clf.predict(self.other[:,1:])  
          
        X = np.vstack((train,self.other))  
         # 将预测结果放到SVM模型中进行训练  
        y = np.vstack((train[:,0].reshape(-1,1), pred_y.reshape(-1,1)))[:,0]
        self.w = np.ones(train_data.shape[0])  
        self.w[len(train):] = self.Cu  
        while self.Cu < self.Cl:  
            print(X.shape,y.shape,self.w.shape)  
            self.clf.fit(X[:,1:],y,sample_weight = self.w)  
            while True:  
                   #返回的是带符号的距离
                dist = self.clf.decision_function(X[:,1:])   
                xi = 1 - y * dist  
                #取出预判为正例和负例的id  
                xi_posi, xi_negi = np.where(y[2:]>0),np.where(y[2:]<0)
                xi_pos , xi_neg = xi[xi_posi],xi[xi_negi]
                xi_pos_maxi = np.argmax(xi_pos)  
                xi_neg_maxi = np.argmax(xi_neg)  
                xi_pos_max = xi_pos[xi_pos_maxi]  
                xi_neg_max = xi_neg[xi_neg_maxi]  
                #不断地拿两个距离最大的点进行交换。
                   #交换策略：两个点中至少有一个误分类。 
                if xi_pos_max >0 and xi_neg_max > 0 \
                     and (xi_pos_max + xi_neg_max) > 2:
                    # 交换类别  
                    y[xi_pos_maxi],y[xi_neg_maxi] = \
                      y[xi_neg_maxi],y[xi_pos_maxi]
                    self.clf.fit(X[:,1:],y, sample_weight = self.w)  
                else:  
                    break  
            self.Cu = min(2 * self.Cu ,self.Cl)  
            # 交换权重  
            self.w[len(train):] = self.Cu  
    def predict(self):
        pred_y = self.clf.predict(self.other[:,1:])
        return 1 - np.mean(pred_y == self.other[:,0])

import numpy as np    
import matplotlib.pyplot as plt    
from sklearn.semi_supervised import label_propagation    
from sklearn.datasets import make_moons  
  
# 生成弧形数据    
n_samples = 200     
X, y  = make_moons(n_samples, noise=0.04, random_state=0)    
outer, inner = 0, 1    
labels = np.full(n_samples, -1.)    
labels[0] = outer    
labels[-1] = inner    
# 使用LP算法实现标签传递   
label_spread = label_propagation.LabelSpreading(kernel='rbf')    
label_spread.fit(X, labels)    
    
# 输出标签    
output_labels = label_spread.transduction_    
plt.figure(figsize=(8.5, 4))    
plt.subplot(1, 2, 1)    
plt.scatter(X[labels == outer, 0],   
            X[labels == outer, 1], color='navy',    
      marker='s', lw=0, label="outer labeled", s=10)    
plt.scatter(X[labels == inner, 0], X[labels == inner, 1],   
            color='c', marker='s', lw=0, label='inner labeled', s=10)    
plt.scatter(X[labels == -1, 0], X[labels == -1, 1],   
            color='darkorange', marker='.', label='unlabeled')    
plt.legend(scatterpoints=1, shadow=False, loc='upper right')    
plt.title("Raw data (2 classes=outer and inner)")    
    
plt.subplot(1, 2, 2)    
output_label_array = np.asarray(output_labels)    
outer_numbers = np.where(output_label_array == outer)[0]    
inner_numbers = np.where(output_label_array == inner)[0]    
plt.scatter(X[outer_numbers, 0], X[outer_numbers, 1], color='navy',    
      marker='s', lw=0, s=10, label="outer learned")    
plt.scatter(X[inner_numbers, 0], X[inner_numbers, 1], color='c',    
      marker='s', lw=0, s=10, label="inner learned")    
plt.legend(scatterpoints=1, shadow=False, loc='upper right')    
plt.title("Labels learned with Label Spreading (KNN)")    
    
plt.subplots_adjust(left=0.07, bottom=0.07, right=0.9, top=0.92)    
plt.show() 
