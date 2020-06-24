# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:08:42 2019

@author: zixing.mei
"""
import math

def sloveKS(self, model, X, Y, Weight):  
    Y_predict = [s[1] for s in model.predict_proba(X)]  
    nrows = X.shape[0]  
    #还原权重  
    lis = [(Y_predict[i], Y.values[i], Weight[i]) for i in range(nrows)]
    #按照预测概率倒序排列  
    ks_lis = sorted(lis, key=lambda x: x[0], reverse=True)        
    KS = list()  
    bad = sum([w for (p, y, w) in ks_lis if y > 0.5])  
    good = sum([w for (p, y, w) in ks_lis if y <= 0.5])  
    bad_cnt, good_cnt = 0, 0  
    for (p, y, w) in ks_lis:  
        if y > 0.5:  
            #1*w 即加权样本个数  
            bad_cnt += w                
        else:  
            #1*w 即加权样本个数  
            good_cnt += w               
        ks = math.fabs((bad_cnt/bad)-(good_cnt/good))  
        KS.append(ks)  
    return max(KS) 

def slovePSI(self, model, dev_x, val_x):  
    dev_predict_y = [s[1] for s in model.predict_proba(dev_x)]  
    dev_nrows = dev_x.shape[0]  
    dev_predict_y.sort()  
    #等频分箱成10份  
    cutpoint = [-100] + [dev_predict_y[int(dev_nrows/10*i)] 
                         for i in range(1, 10)] + [100]  
    cutpoint = list(set(cutpoint))  
    cutpoint.sort()
    val_predict_y = [s[1] for s in list(model.predict_proba(val_x))]  
    val_nrows = val_x.shape[0]  
    PSI = 0  
    #每一箱之间分别计算PSI  
    for i in range(len(cutpoint)-1):  
        start_point, end_point = cutpoint[i], cutpoint[i+1]  
        dev_cnt = [p for p in dev_predict_y 
                                 if start_point <= p < end_point]  
        dev_ratio = len(dev_cnt) / dev_nrows + 1e-10  
        val_cnt = [p for p in val_predict_y 
                                 if start_point <= p < end_point]  
        val_ratio = len(val_cnt) / val_nrows + 1e-10  
        psi = (dev_ratio - val_ratio) * math.log(dev_ratio/val_ratio)
        PSI += psi  
    return PSI  

import xgboost as xgb  
from xgboost import plot_importance  
  
class xgBoost(object):  
    def __init__(self, datasets, uid, dep, weight, 
                                  var_names, params, max_del_var_nums=0):
        self.datasets = datasets  
        #样本唯一标识，不参与建模  
        self.uid = uid       
        #二分类标签  
        self.dep = dep     
        #样本权重  
        self.weight = weight      
        #特征列表  
        self.var_names = var_names    
        #参数字典，未指定字段使用默认值  
        self.params = params     
        #单次迭代最多删除特征的个数  
        self.max_del_var_nums = max_del_var_nums    
        self.row_num = 0  
        self.col_num = 0  
  
    def training(self, min_score=0.0001, modelfile="", output_scores=list()):  
        lis = self.var_names[:]  
        dev_data = self.datasets.get("dev", "")  #训练集  
        val_data = self.datasets.get("val", "")  #测试集  
        off_data = self.datasets.get("off", "")  #跨时间验证集
                #从字典中查找参数值，没有则使用第二项作为默认值  
        model = xgb.XGBClassifier(
                           learning_rate=self.params.get("learning_rate", 0.1),
              n_estimators=self.params.get("n_estimators", 100),  
              max_depth=self.params.get("max_depth", 3),  
              min_child_weight=self.params.get("min_child_weight", 1),subsample=self.params.get("subsample", 1),  
              objective=self.params.get("objective", 
                                                             "binary:logistic"),
              nthread=self.params.get("nthread", 10),  
              scale_pos_weight=self.params.get("scale_pos_weight", 1),
              random_state=0,  
              n_jobs=self.params.get("n_jobs", 10),  
              reg_lambda=self.params.get("reg_lambda", 1),  
              missing=self.params.get("missing", None) )  
        while len(lis) > 0:   
            #模型训练  
            model.fit(X=dev_data[self.var_names], y=dev_data[self.dep])  
            #得到特征重要性  
            scores = model.feature_importances_     
            #清空字典  
            lis.clear()      
            ''' 
            当特征重要性小于预设值时， 
            将特征放入待删除列表。 
            当列表长度超过预设最大值时，跳出循环。 
            即一次只删除限定个数的特征。 
            '''  
            for (idx, var_name) in enumerate(self.var_names):  
                #小于特征重要性预设值则放入列表  
                if scores[idx] < min_score:    
                    lis.append(var_name)  
                #达到预设单次最大特征删除个数则停止本次循环  
                if len(lis) >= self.max_del_var_nums:     
                    break  
            #训练集KS  
            devks = self.sloveKS(model, dev_data[self.var_names],
                                       dev_data[self.dep], dev_data[self.weight])
            #初始化ks值和PSI  
            valks, offks, valpsi, offpsi = 0.0, 0.0, 0.0, 0.0 
            #测试集KS和PSI  
            if not isinstance(val_data, str):  
                valks = self.sloveKS(model,
                                                      val_data[self.var_names], 
                                                      val_data[self.dep], 
                                                      val_data[self.weight])  
                valpsi = self.slovePSI(model,
                                                        dev_data[self.var_names],
                                                        val_data[self.var_names])
            #跨时间验证集KS和PSI  
            if not isinstance(off_data, str):  
                offks = self.sloveKS(model,
                                                  off_data[self.var_names],
                                                  off_data[self.dep],
                                                  off_data[self.weight])  
                offpsi = self.slovePSI(model,
                                                     dev_data[self.var_names],
                                                     off_data[self.var_names])  
            #将三个数据集的KS和PSI放入字典  
            dic = {"devks": float(devks), 
                                 "valks": float(valks),
                                  "offks": offks,  
                 "valpsi": float(valpsi),
                                  "offpsi": offpsi}  
            print("del var: ", len(self.var_names), 
                                       "-->", len(self.var_names) - len(lis),
                                       "ks: ", dic, ",".join(lis))
            self.var_names = [var_name for var_name in self.var_names if var_name not in lis]
        plot_importance(model)  
        #重新训练，准备进入下一循环  
        model = xgb.XGBClassifier(
                             learning_rate=self.params.get("learning_rate", 0.1),
               n_estimators=self.params.get("n_estimators", 100),
                 max_depth=self.params.get("max_depth", 3),  
                 min_child_weight=self.params.get("min_child_weight",1),
               subsample=self.params.get("subsample", 1),  
               objective=self.params.get("objective", 
                                                        "binary:logistic"),  
               nthread=self.params.get("nthread", 10),  
               scale_pos_weight=self.params.get("scale_pos_weight",1),
               random_state=0,  
               n_jobs=self.params.get("n_jobs", 10),  
               reg_lambda=self.params.get("reg_lambda", 1),  
               missing=self.params.get("missing", None))  








