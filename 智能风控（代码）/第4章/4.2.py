# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:50:39 2019

@author: zixing.mei
"""

import xgboost as xgb  
from sklearn.datasets import load_digits # 训练数据  
xgb_params_01 = {}  
digits_2class = load_digits(2)  
X_2class = digits_2class['data']  
y_2class = digits_2class['target']  
dtrain_2class = xgb.DMatrix(X_2class, label=y_2class)
# 训练三棵树的模型  
gbdt_03 = xgb.train(xgb_params_01, dtrain_2class, num_boost_round=3) 
# 以前面三棵树的模型为基础，从第四棵树开始训练  
gbdt_03a = xgb.train(xgb_params_01, dtrain_2class, num_boost_round=7, xgb_model=gbdt_03)  


