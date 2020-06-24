# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:57:09 2019

@author: zixing.mei
"""

import lightgbm as lgb  
import random  
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error  
from sklearn.linear_model import LogisticRegression  
from sklearn import metrics  
from sklearn.metrics import roc_curve  
from matplotlib import pyplot as plt  
import math  
  
df_train = data[data.obs_mth != '2018-11-30'].reset_index().copy()    
df_test = data[data.obs_mth == '2018-11-30'].reset_index().copy()    
NUMERIC_COLS = ['person_info','finance_info','credit_info','act_info']
from sklearn.preprocessing import OneHotEncoder,LabelEncoder  
  
lgb_train = lgb.Dataset(df_train[NUMERIC_COLS], 
                          df_train['bad_ind'], free_raw_data=False)  
params = {  
    'num_boost_round': 50,  
    'boosting_type': 'gbdt',  
    'objective': 'binary',  
    'num_leaves': 2,  
    'metric': 'auc',  
    'max_depth':1,  
    'feature_fraction':1,  
    'bagging_fraction':1, } 
model = lgb.train(params,lgb_train)  
leaf = model.predict(df_train[NUMERIC_COLS],pred_leaf=True)  
lgb_enc = OneHotEncoder()  
#生成交叉特征
lgb_enc.fit(leaf)
#和原始特征进行合并
data_leaf = np.hstack((lgb_enc.transform(leaf).toarray(),df_train[NUMERIC_COLS]))  
leaf_test = model.predict(df_test[NUMERIC_COLS],pred_leaf=True)  
lgb_enc = OneHotEncoder()  
lgb_enc.fit(leaf_test)  
data_leaf_test = np.hstack((lgb_enc.transform(leaf_test).toarray(),
                              df_test[NUMERIC_COLS]))  
train = data_leaf.copy()  
train_y = df_train['bad_ind'].copy()  
val = data_leaf_test.copy()  
val_y = df_test['bad_ind'].copy()  
lgb_lm = LogisticRegression(penalty='l2',C=0.2, class_weight='balanced',solver='liblinear')
lgb_lm.fit(train, train_y)  
y_pred_lgb_lm_train = lgb_lm.predict_proba(train)[:, 1]  
fpr_lgb_lm_train, tpr_lgb_lm_train, _ = roc_curve(train_y,y_pred_lgb_lm_train)
y_pred_lgb_lm = lgb_lm.predict_proba(val)[:,1]  
fpr_lgb_lm,tpr_lgb_lm,_ = roc_curve(val_y,y_pred_lgb_lm)  
plt.figure(1)  
plt.plot([0, 1], [0, 1], 'k--')  
plt.plot(fpr_lgb_lm_train,tpr_lgb_lm_train,label='LGB + LR train')  
plt.plot(fpr_lgb_lm, tpr_lgb_lm, label='LGB + LR test')  
plt.xlabel('False positive rate')  
plt.ylabel('True positive rate')  
plt.title('ROC curve')  
plt.legend(loc='best')  
plt.show()  
print('LGB+LR train ks:',abs(fpr_lgb_lm_train - tpr_lgb_lm_train).max(),
                               'LGB+LR AUC:', metrics.auc(fpr_lgb_lm_train, tpr_lgb_lm_train))
print('LGB+LR test ks:',abs(fpr_lgb_lm - tpr_lgb_lm).max(),
                              'LGB+LR AUC:', metrics.auc(fpr_lgb_lm, tpr_lgb_lm))
dff_train = pd.DataFrame(train)  
dff_train.columns = [ 'ft' + str(x) for x in range(train.shape[1])]  
  
dff_val = pd.DataFrame(val)  
dff_val.columns = [ 'ft' + str(x) for x in range(val.shape[1])]  
#生成可以传入PSI的数据集  
def make_psi_data(dff_train):  
    dftot = pd.DataFrame()  
    for col in dff_train.columns:  
        zero= sum(dff_train[col] == 0)  
        one= sum(dff_train[col] == 1)  
        ftdf = pd.DataFrame(np.array([zero,one]))  
        ftdf.columns = [col]  
        if len(dftot) == 0:  
            dftot = ftdf.copy()  
        else:  
            dftot[col] = ftdf[col].copy()  
    return dftot  
psi_data_train = make_psi_data(dff_train)  
psi_data_val = make_psi_data(dff_val) 
def var_PSI(dev_data, val_data):  
    dev_cnt, val_cnt = sum(dev_data), sum(val_data)  
    if dev_cnt * val_cnt == 0:  
        return 0  
    PSI = 0  
    for i in range(len(dev_data)):  
        dev_ratio = dev_data[i] / dev_cnt  
        val_ratio = val_data[i] / val_cnt + 1e-10  
        psi = (dev_ratio - val_ratio) * math.log(dev_ratio/val_ratio)
        PSI += psi  
    return PSI  
psi_dct = {}  
for col in dff_train.columns:  
    psi_dct[col] = var_PSI(psi_data_train[col],psi_data_val[col]) 
f = zip(psi_dct.keys(),psi_dct.values())  
f = sorted(f,key = lambda x:x[1],reverse = False)  
psi_df = pd.DataFrame(f)  
psi_df.columns = pd.Series(['变量名','PSI'])  
feature_lst = list(psi_df[psi_df['PSI']<psi_df.quantile(0.6)[0]]['变量名'])  
train = dff_train[feature_lst].copy()  
train_y = df_train['bad_ind'].copy()  
val = dff_val[feature_lst].copy()  
val_y = df_test['bad_ind'].copy()  
lgb_lm = LogisticRegression(C = 0.3,class_weight='balanced',solver='liblinear')
lgb_lm.fit(train, train_y)  
y_pred_lgb_lm_train = lgb_lm.predict_proba(train)[:, 1]  
fpr_lgb_lm_train, tpr_lgb_lm_train, _ = roc_curve(train_y, y_pred_lgb_lm_train)
y_pred_lgb_lm = lgb_lm.predict_proba(val)[:, 1]  
fpr_lgb_lm, tpr_lgb_lm, _ = roc_curve(val_y, y_pred_lgb_lm)  
plt.figure(1)  
plt.plot([0, 1], [0, 1], 'k--')  
plt.plot(fpr_lgb_lm_train, tpr_lgb_lm_train, label='LGB + LR train')  
plt.plot(fpr_lgb_lm, tpr_lgb_lm, label='LGB + LR test')  
plt.xlabel('False positive rate')  
plt.ylabel('True positive rate')  
plt.title('ROC curve')  
plt.legend(loc='best')  
plt.show()  
print('LGB+LR train ks:',abs(fpr_lgb_lm_train - tpr_lgb_lm_train).max(),
                               'LGB+LR AUC:', metrics.auc(fpr_lgb_lm_train, tpr_lgb_lm_train))
print('LGB+LR test ks:',abs(fpr_lgb_lm - tpr_lgb_lm).max(),'LGB+LR AUC:',
                              metrics.auc(fpr_lgb_lm, tpr_lgb_lm))
x = train  
y = train_y  
  
val_x =  val  
val_y = val_y  
  
#定义lgb函数  
def LGB_test(train_x,train_y,test_x,test_y):  
    from multiprocessing import cpu_count  
    clf = lgb.LGBMClassifier(  
        boosting_type='gbdt', num_leaves=31, reg_Ap=0.0, reg_lambda=1,
        max_depth=2, n_estimators=800,max_features=140,objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,  
        learning_rate=0.05, min_child_weight=50,
              random_state=None,n_jobs=cpu_count()-1,)  
    clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(test_x,test_y)],
                eval_metric='auc',early_stopping_rounds=100)  
    return clf,clf.best_score_[ 'valid_1']['auc']  
#训练模型
model,auc = LGB_test(x,y,val_x,val_y)                      
  
#模型贡献度放在feture中  
feature = pd.DataFrame(  
            {'name' : model.booster_.feature_name(),  
            'importance' : model.feature_importances_  
          }).sort_values(by = ['importance'],ascending = False) 
feature_lst2 = list(feature[feature.importance>5].name)
train = dff_train[feature_lst2].copy()  
train_y = df_train['bad_ind'].copy()  
val = dff_val[feature_lst2].copy()  
val_y = df_test['bad_ind'].copy()  
lgb_lm = LogisticRegression(C = 0.3,class_weight='balanced',solver='liblinear')
lgb_lm.fit(train, train_y)  
  
y_pred_lgb_lm_train = lgb_lm.predict_proba(train)[:, 1]  
fpr_lgb_lm_train, tpr_lgb_lm_train, _ = roc_curve(train_y, y_pred_lgb_lm_train)
  
y_pred_lgb_lm = lgb_lm.predict_proba(val)[:, 1]  
fpr_lgb_lm, tpr_lgb_lm, _ = roc_curve(val_y, y_pred_lgb_lm)  
  
plt.figure(1)  
plt.plot([0, 1], [0, 1], 'k--')  
plt.plot(fpr_lgb_lm_train, tpr_lgb_lm_train, label='LGB + LR train')  
plt.plot(fpr_lgb_lm, tpr_lgb_lm, label='LGB + LR test')  
plt.xlabel('False positive rate')  
plt.ylabel('True positive rate')  
plt.title('ROC curve')  
plt.legend(loc='best')  
plt.show()  
print('LGB+LR train ks:',abs(fpr_lgb_lm_train - tpr_lgb_lm_train).max(),
      'LGB+LR AUC:', metrics.auc(fpr_lgb_lm_train, tpr_lgb_lm_train))  
print('LGB+LR test ks:',abs(fpr_lgb_lm - tpr_lgb_lm).max(),'LGB+LR AUC:', 
      metrics.auc(fpr_lgb_lm, tpr_lgb_lm))  

