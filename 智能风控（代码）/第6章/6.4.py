# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:54:36 2019

@author: zixing.mei
"""

from pyod.models.iforest import IForest
clf = IForest(behaviour='new', bootstrap=False, contamination=0.1, max_features=1.0,
                max_samples='auto', n_estimators=500, n_jobs=-1, random_state=None,verbose=0)
clf.fit(x)
out_pred = clf.predict_proba(x,method ='linear')[:,1]
train['out_pred'] = out_pred
train['for_pred'] = np.where(train.out_pred>0.7,'负样本占比','正样本占比')
dic = dict(train.groupby(train.for_pred).bad_ind.agg(np.sum)/ \
           train.bad_ind.groupby(train.for_pred).count())
pd.DataFrame(dic,index=[0])

clf = IForest(behaviour='new', bootstrap=False, contamination=0.1, max_features=1.0,
                max_samples='auto', n_estimators=500, n_jobs=-1, random_state=None,verbose=0)
clf.fit(x)
y_pred = clf.predict_proba(x,method ='linear')[:,1]    
fpr_lr_train,tpr_lr_train,_ = roc_curve(y,y_pred)    
train_ks = abs(fpr_lr_train - tpr_lr_train).max()    
print('train_ks : ',train_ks)    
y_pred = clf.predict_proba(val_x,method ='linear')[:,1]    
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

