# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:13:32 2019

@author: zixing.mei
"""

def auto_delete_vars(self):  
    dev_data = self.datasets.get("dev", "")  
    off_data = self.datasets.get("off", "")  
    params = self.params  
    model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),  
                 learning_rate=params.get("learning_rate", 0.05),
                 n_estimators=params.get("n_estimators", 100),
                 min_child_weight=params.get("min_child_weight",1),
                  subsample=params.get("subsample", 1),  
                  scale_pos_weight=params.get("scale_pos_weight",1),
                 reg_lambda=params.get("reg_lambda", 1),  
                 nthread=8, n_jobs=8, random_state=7)  
    model.fit(dev_data[self.var_names], 
                  dev_data[self.dep], dev_data[self.weight])  
    offks = self.sloveKS(model, off_data[self.var_names], 
                               off_data[self.dep], off_data[self.weight])  
    train_number = 0  
    print("train_number: %s, offks: %s" % (train_number, offks))  
    del_list = list()  
    oldks = offks  
    while True:  
        bad_ind = True  
        for var_name in self.var_names:  
            #遍历每一个特征  
            model=xgb.XGBClassifier(
                                  max_depth=params.get("max_depth", 3),  
                 learning_rate=params.get("learning_rate",0.05),
                 n_estimators=params.get("n_estimators", 100), 
                 min_child_weight=params.get("min_child_weight",1),
                 subsample=params.get("subsample", 1),  
                 scale_pos_weight=params.get("scale_pos_weight",1),
                 reg_lambda=params.get("reg_lambda", 1),  
                 nthread=10,n_jobs=10,random_state=7)  
            #将当前特征从模型中去掉  
            names = [var for var in self.var_names 
                                    if var_name != var]  
            model.fit(dev_data[names], dev_data[self.dep], 
                                  dev_data[self.weight])  
            train_number += 1  
            offks = self.sloveKS(model, off_data[names], 
                                     off_data[self.dep], off_data[self.weight])
            ''' 
            比较KS是否有提升， 
            如果有提升或者武明显变化， 
            则可以将特征去掉 
            '''  
            if offks >= oldks:  
                oldks = offks  
                bad_ind = False  
                del_list.append(var_name)  
                self.var_names = names  
            else:  
                continue
        if bad_ind:  
            break  
    print("(End) train_n: %s, offks: %s del_list_vars: %s" 
                  % (train_number, offks, del_list))  

