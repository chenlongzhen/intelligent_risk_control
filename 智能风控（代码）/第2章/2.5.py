# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:11:59 2019

@author: zixing.mei
"""

def target_value(self,old_devks,old_offks,target,devks,offks,w=0.2):  
    '''  
    如果参数设置为"best"，使用最优调参策略， 
    否则使用跨时间测试集KS最大策略。 
    '''  
    if target == "best":  
        return offks-abs(devks-offks)*w
    else:  
        return offks  

def check_params(self, dev_data, off_data, params, param, train_number, step, target, 
                                                            targetks, old_devks, old_offks):  
    ''' 
    当前向搜索对调参策略有提升时， 
    继续前向搜索。 
    否则进行后向搜索 
    '''  
    while True:  
        try:  
            if params[param] + step > 0:  
                params[param] += step  
                model = xgb.XGBClassifier(
                                   max_depth=params.get("max_depth", 3),
                                   learning_rate=params.get("learning_rate", 0.05),
                                   n_estimators=params.get("n_estimators", 100),
                                   min_child_weight=params.get(
                                                       "min_child_weight", 1),
                                   subsample=params.get("subsample", 1),  
                                   scale_pos_weight=params.get(
                                   "scale_pos_weight", 1),
                                   nthread=10,n_jobs=10, random_state=0)  
                model.fit(dev_data[self.var_names],
                                              dev_data[self.dep],
                                              dev_data[self.weight])  
                devks = self.sloveKS(model, 
                                                       dev_data[self.var_names], 
                                                       dev_data[self.dep], 
                                                       dev_data[self.weight])  
                offks = self.sloveKS(model, 
                                                       off_data[self.var_names], 
                                                       off_data[self.dep], 
                                                       off_data[self.weight])  
                train_number += 1  
                targetks_n = self.target_value(
                                                      old_devks=old_devks, 
                                                      old_offks=old_offks, 
                                                      target=target,  
                                                      devks=devks, 
                                                      offks=offks)  
                if targetks < targetks_n:  
                    targetks = targetks_n  
                    old_devks = devks  
                    old_offks = offks  
                else:  
                    break  
            else:  
                break  
        except:  
            break  
    params[param] -= step  
    return params, targetks, train_number  

def auto_choose_params(self, target="offks"):  
    """ 
    "mzh1": offks + (offks - devks) * 0.2 最大化   
        "mzh2": (offks + (offks - devks) * 0.2)**2 最大化 
        其余取值均使用跨时间测试集offks  最大化
    当业务稳定性较差时，应将0.2改为更大的值 
    """  
    dev_data = self.datasets.get("dev", "")  
    off_data = self.datasets.get("off", "")  
    #设置参数初始位置  
    params = {  
        "max_depth": 5,  
        "learning_rate": 0.09,  
        "n_estimators": 120,  
        "min_child_weight": 50,  
        "subsample": 1,  
        "scale_pos_weight": 1,  
        "reg_lambda": 21  
    }  
    model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),  
                                  learning_rate=params.get("learning_rate", 0.05),
                 n_estimators=params.get("n_estimators", 100),
                 min_child_weight=params.get("min_child_weight",1),
                 subsample=params.get("subsample", 1),
                 scale_pos_weight=params.get("scale_pos_weight",1),
                 reg_lambda=params.get("reg_lambda", 1),
                 nthread=8, n_jobs=8, random_state=7)  
    model.fit(dev_data[self.var_names], 
                      dev_data[self.dep],
                      dev_data[self.weight])  
    devks = self.sloveKS(model, 
                               dev_data[self.var_names], 
                               dev_data[self.dep], 
                               dev_data[self.weight])  
    offks = self.sloveKS(model,
                                    off_data[self.var_names], 
                                    off_data[self.dep], 
                                    off_data[self.weight])  
    train_number = 0  
    #设置调参步长  
    dic = {  
        "learning_rate": [0.05, -0.05],  
        "max_depth": [1, -1],  
        "n_estimators": [20, 5, -5, -20],  
        "min_child_weight": [20, 5, -5, -20],  
        "subsample": [0.05, -0.05],  
        "scale_pos_weight": [20, 5, -5, -20],  
        "reg_lambda": [10, -10]  
    }  
    #启用调参策略  
    targetks = self.target_value(old_devks=devks, 
                                       old_offks=offks, target=target, 
                                       devks=devks, offks=offks)  
    old_devks = devks  
    old_offks = offks  
    #按照参数字典，双向搜索最优参数  
    while True:  
        targetks_lis = []  
        for (key, values) in dic.items():  
            for v in values:  
                if v + params[key] > 0:  
                    params, targetks, train_number = \
                                                       self.check_params(dev_data, 
                                                       off_data, params, 
                                                       key, train_number,  
                            v, target, targetks, 
                                                       old_devks, old_offks)  
                    targetks_n = self.target_value(
                                                         old_devks=old_devks, 
                                                         old_offks=old_offks, 
                                                         target=target,  
                             devks=devks, offks=offks)
                    if targetks < targetks_n:  
                        old_devks = devks  
                        old_offks = offks  
                        targetks_lis.append(targetks)  
        if not targetks_lis:  
            break  
    print("Best params: ", params)  
    model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),  
                   learning_rate=params.get("learning_rate", 0.05),
                  n_estimators=params.get("n_estimators", 100),
                 min_child_weight=params.get("min_child_weight",1),
                 subsample=params.get("subsample", 1),  
                 scale_pos_weight=params.get("scale_pos_weight",1),
                 reg_lambda=params.get("reg_lambda", 1),  
                 nthread=10, n_jobs=10, random_state=0)  
    model.fit(dev_data[self.var_names], 
                  dev_data[self.dep], dev_data[self.weight])  

