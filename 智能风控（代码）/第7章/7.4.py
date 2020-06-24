# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:01:35 2019

@author: zixing.mei
"""

#加载xlearn包    
import xlearn as xl    
#调用FM模型    
fm_model = xl.create_fm()    
# 训练集    
fm_model.setTrain("train.txt")    
# 设置验证集    
fm_model.setValidate("test.txt")   
# 分类问题：acc(Accuracy);prec(precision);f1(f1 score);auc(AUC score)    
param = {'task':'binary','lr':0.2,'lambda':0.002,'metric':'auc'}    
fm_model.fit(param, "model.out")   
fm_model.setSigmoid()    
fm_model.predict("model.out","output.txt")    
fm_model.setTXTModel("model.txt")  

