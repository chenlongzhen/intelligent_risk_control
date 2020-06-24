# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:02:19 2019

@author: zixing.mei
"""

from heamy.dataset import Dataset  
from heamy.estimator import Regressor  
from heamy.pipeline import ModelsPipeline  
import pandas as pd  
import xgboost as xgb  
from sklearn.metrics import roc_auc_score  
import lightgbm as lgb  
from sklearn.linear_model import LinearRegression  
from sklearn.ensemble import ExtraTreesClassifier  
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn import svm  
import numpy as np  
  
def xgb_model1(X_train, y_train, X_test, y_test=None):  
    # xgboost1  
    params = {'booster': 'gbtree',  
              'objective':'rank:pairwise',  
              'eval_metric' : 'auc',  
              'eta': 0.02,  
              'max_depth': 5,  # 4 3  
              'colsample_bytree': 0.7,#0.8  
              'subsample': 0.7,  
              'min_child_weight': 1,  # 2 3  
              'seed': 1111,  
              'silent':1  
              }  
    dtrain = xgb.DMatrix(X_train, label=y_train)  
    dvali = xgb.DMatrix(X_test)  
    model = xgb.train(params, dtrain, num_boost_round=800)  
    predict = model.predict_proba(dvali)  
    minmin = min(predict)  
    maxmax = max(predict)  
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
    return vfunc(predict)  
  
def xgb_model2(X_train, y_train, X_test, y_test=None):  
    # xgboost2  
    params = {'booster': 'gbtree',  
              'objective':'rank:pairwise',  
              'eval_metric' : 'auc',  
              'eta': 0.015,  
              'max_depth': 5,  # 4 3  
              'colsample_bytree': 0.7,#0.8  
              'subsample': 0.7,  
              'min_child_weight': 1,  # 2 3  
              'seed': 11,  
              'silent':1  
              }  
    dtrain = xgb.DMatrix(X_train, label=y_train)  
    dvali = xgb.DMatrix(X_test)  
    model = xgb.train(params, dtrain, num_boost_round=1200)  
    predict = model.predict_proba (dvali)  
    minmin = min(predict)  
    maxmax = max(predict)  
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
    return vfunc(predict)  
  
def xgb_model3(X_train, y_train, X_test, y_test=None):  
    # xgboost3  
    params = {'booster': 'gbtree',  
              'objective':'rank:pairwise',  
              'eval_metric' : 'auc',  
              'eta': 0.01,  
              'max_depth': 5,  # 4 3  
              'colsample_bytree': 0.7,#0.8  
              'subsample': 0.7,  
              'min_child_weight': 1,  # 2 3  
              'seed': 1,  
              'silent':1  
              }  
    dtrain = xgb.DMatrix(X_train, label=y_train)  
    dvali = xgb.DMatrix(X_test)  
    model = xgb.train(params, dtrain, num_boost_round=2000)  
    predict = model.predict_proba (dvali)  
    minmin = min(predict)  
    maxmax = max(predict)  
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
    return vfunc(predict)  
  
def et_model(X_train, y_train, X_test, y_test=None):  
    #ExtraTree  
    model = ExtraTreesClassifier(max_features='log2',n_estimators=1000,n_jobs=1).fit(X_train,y_train)
    predict = model.predict_proba(X_test)[:,1] 
    minmin = min(predict)  
    maxmax = max(predict)  
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
    return vfunc(predict)     
  
def gbdt_model(X_train, y_train, X_test, y_test=None):  
    #GBDT  
    model = GradientBoostingClassifier(learning_rate=0.02,max_features=0.7,
                                             n_estimators=700,max_depth=5).fit(X_train,y_train)
    predict = model.predict_proba(X_test)[:,1]  
    minmin = min(predict)  
    maxmax = max(predict)  
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
    return vfunc(predict)  
  
def logistic_model(X_train, y_train, X_test, y_test=None):  
    #逻辑回归  
    model = LogisticRegression(penalty = 'l2').fit(X_train,y_train)  
    predict = model.predict_proba(X_test)[:,1] 
    minmin = min(predict)  
    maxmax = max(predict)  
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
    return vfunc(predict)  
 
  
def lgb_model(X_train, y_train, X_test, y_test=None):  
    #LightGBM  
    lgb_train=lgb.Dataset(X_train,y_train,categorical_feature={'sex','merriage','income',
                                                                   'qq_bound','degree',
                                                                   'wechat_bound',
                                                                   'account_grade','industry'})
    lgb_test = lgb.Dataset(X_test,categorical_feature={'sex','merriage','income','qq_bound',
                                                             'degree','wechat_bound',
                                                             'account_grade','industry'})  
    params = {  
        'task': 'train',  
        'boosting_type': 'gbdt',  
        'objective': 'binary',  
        'metric':'auc',  
        'num_leaves': 25,  
        'learning_rate': 0.01,  
        'feature_fraction': 0.7,  
        'bagging_fraction': 0.7,  
        'bagging_freq': 5,  
        'min_data_in_leaf':5,  
        'max_bin':200,  
        'verbose': 0,  
    }  
    gbm = lgb.train(params,  
    lgb_train,  
    num_boost_round=2000)  
    predict = gbm.predict_proba(X_test)  
    minmin = min(predict)  
    maxmax = max(predict)  
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
    return vfunc(predict)  
  
def svm_model(X_train, y_train, X_test, y_test=None):  
    #支持向量机  
    model = svm.SVC(C=0.8,kernel='rbf',gamma=20,
                          decision_function_shape='ovr').fit(X_train,y_train)
    predict = model.predict_proba(X_test)[:,1]  
   	minmin = min(predict)  
   	maxmax = max(predict)  
  	vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))  
  	return vfunc(predict)  
  
import pandas as pd  
import numpy as np  
from minepy import MINE  
  
""" 
  从csv文件中，加载8个模型的预测分数 
"""  
xgb1_result = pd.read_csv('xgb1.csv')  
xgb2_result = pd.read_csv('xgb2.csv')  
xgb3_result = pd.read_csv('xgb3.csv')  
et_result = pd.read_csv('et_model.csv')  
svm_result = pd.read_csv('svm.csv')  
lr_result = pd.read_csv('lr.csv')  
lgb_result = pd.read_csv('lgb.csv')  
gbdt_result = pd.read_csv('gbdt.csv')  
  
res = []  
res.append(xgb1_result.score.values)  
res.append(xgb2_result.score.values)  
res.append(xgb3_result.score.values)  
res.append(et_result.score.values)  
res.append(svm_result.score.values)  
res.append(lr_result.score.values)  
res.append(lgb_result.score.values)  
res.append(gbdt_result.score.values)  
  
""" 
  计算向量两两之间的MIC值 
"""  
cm = []  
for i in range(7):  
    tmp = []  
    for j in range(7):  
        m = MINE()  
        m.compute_score(res[i], res[j])  
        tmp.append(m.mic())  
    cm.append(tmp)  
  
""" 
    绘制MIC图像 
"""  
  
fs = ['xgb1','xgb2','xgb3','et','svm','lr','lgb','gbdt']  
  
import matplotlib.pyplot as plt  
  
def plot_confusion_matrix(cm, title, cmap=plt.cm.Blues):  
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  
    plt.title(title)  
    plt.colorbar()  
    tick_marks = np.arange(8)  
    plt.xticks(tick_marks, fs, rotation=45)  
    plt.yticks(tick_marks, fs)  
    plt.tight_layout()  
  
plot_confusion_matrix(cm, title='mic')  
plt.show() 
model_xgb2 = Regressor(dataset= dataset, estimator=xgb_feature2,name='xgb2',use_cache=False) 
model_lr = Regressor(dataset= dataset, estimator=logistic_model,name='lr',use_cache=False) 
model_lgb = Regressor(dataset= dataset, estimator=lgb_model,name='lgb',use_cache=False)  
model_ gbdt = Regressor(dataset= dataset, estimator=gbdt_model,name='gbdt',use_cache=False)
pipeline = ModelsPipeline(model_xgb2, model_lr, model_lgb, model_svm)  
stack_data = pipeline.stack(k=5, seed=0, add_diff=False, full_test=True)  
stacker = Regressor(dataset=stack_data,estimator=LinearRegression,
                      parameters={'fit_intercept': False})  
predict_result = stacker.predict()
val = pd.read_csv('val_list.csv')
val['PROB'] = predict_result
minmin, maxmax = min(val ['PROB']),max(val ['PROB'])
val['PROB'] = val['PROB'].map(lambda x:(x-minmin)/(maxmax-minmin))
val['PROB'] = val['PROB'].map(lambda x:'%.4f' % x)

