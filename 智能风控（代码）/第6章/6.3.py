# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:51:40 2019

@author: zixing.mei
"""

from pyod.models.lof import LOF    
  
 #训练异常检测模型，然后输出训练集样本的异常分  
clf = LOF(n_neighbors=20, algorithm='auto', leaf_size=30, 
            metric='minkowski', p=2,metric_params=None, 
            contamination=0.1, n_jobs=1)  
clf.fit(x)   
  
#异常分  
out_pred = clf.predict_proba(x,method ='linear')[:,1]    
train['out_pred'] = out_pred    
  
#异常分在0.9百分位以下的样本删掉   
key = train['out_pred'].quantile(0.9)

x = train[train.out_pred< key][feature_lst]
y = train[train.out_pred < key]['bad_ind']   
   
val_x = val[feature_lst]    
val_y = val['bad_ind']    
  
#重新训练模型   
lr_model = LogisticRegression(C=0.1,class_weight='balanced')    
lr_model.fit(x,y)    
y_pred = lr_model.predict_proba(x)[:,1]    
fpr_lr_train,tpr_lr_train,_ = roc_curve(y,y_pred)    
train_ks = abs(fpr_lr_train - tpr_lr_train).max()    
print('train_ks : ',train_ks)    
    
y_pred = lr_model.predict_proba(val_x)[:,1]    
fpr_lr,tpr_lr,_ = roc_curve(val_y,y_pred)    
val_ks = abs(fpr_lr - tpr_lr).max()    
print('val_ks : ',val_ks)    
  
from matplotlib import pyplot as plt    
plt.plot(fpr_lr_train,tpr_lr_train,label = 'train LR')    
plt.plot(fpr_lr,tpr_lr,label = 'evl LR')    
plt.plot([0,1],[0,1],'k--')    
plt.xlabel('False positive rate')    
plt.ylabel('True positive rate')    
plt.title('ROC Curve')    
plt.legend(loc = 'best')    
plt.show()  

