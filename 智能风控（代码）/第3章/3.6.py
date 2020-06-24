# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:32:43 2019

@author: zixing.mei
"""

from sklearn.metrics import roc_auc_score as AUC  
import pandas as pd  
import numpy as np  
   
class Tra_learn3ft (object):  
    """ 
        一种多模型融合的Tradaboost变体 
        使用三个模型同时进行样本筛选，目的是减小variance 
        clfA 模型A 
        clfB 模型B 
        clfC 模型C 
        step 预计去掉的样本比例 
        max_turns最大迭代次数 
    """  
    def __init__(self,clfA,clfB,clfC,step,max_turns=5):  
        self.clfA = clfA  
        self.clfB = clfB  
        self.clfC = clfC  
        self.step = step  
        self.max_turns = max_turns  
        self.scoreA = 0  
        self.scoreB = 0  
        self.scoreC = 0  
  
    def tlearn(self,dev,test,val,bad_ind,featureA,featureB,featureC,drop_rate):  
        """ 
            dev 训练集 源域 
            test 测试集 辅助域 
            val 验证集 
            bad_ind 标签 
            featureA 特征组A 
            featureB 特征组B 
            featureC 特征组C 
        """  
        print(len(featureA),len(featureB),len(featureC))  
        result = pd.DataFrame()  
        temp_test = test  
        features = list(set(featureA+featureB+featureC))  
        turn = 1  
        while( turn <= self.max_turns):  
            new = pd.DataFrame()  
              
            """ 
                模型A对特征组featureA训练， 
                并预测得到dev和test和val的概率 
                以及test上的分类结果（分数分布在0.8*(min+max)两侧）
            """  
            self.clfA.fit(dev[featureA],dev[bad_ind])  
            predA= self.clfA.predict_proba(dev[featureA])[:,1]   
            probA = self.clfA.predict_proba(test[featureA])[:,1]  
            preA = (probA > (np.max(probA)+np.min(probA))*0.8)  
            valid_a = self.clfA.predict_proba(val[featureA])[:,1]   
            """ 
                模型B对特征组featureB训练， 
                并预测得到dev和test和val的概率 
                以及test上的分类结果（分数分布在0.8*(min+max)两侧）
            """  
            self.clfB.fit(dev[featureB],dev[bad_ind])  
            predB = self.clfB.predict_proba(dev[featureB])[:,1]  
            probB = self.clfB.predict_proba(test[featureB])[:,1]  
            preB = (probA > (np.max(probB)+np.min(probB))*0.8)  
            valid_b = self.clfB.predict_proba(val[featureB])[:,1]  
            """ 
                模型C对特征组featureC训练， 
                并预测得到dev和test和val的概率 
                以及test上的分类结果（分数分布在0.8*(min+max)两侧） 
            """              
            self.clfC.fit(dev[featureC],dev[bad_ind])  
            predC= self.clfC.predict_proba(dev[featureC])[:,1]  
            probC = self.clfC.predict_proba(test[featureC])[:,1]  
            preC = (probC > (np.max(probC)+np.min(probC))*0.8)  
            valid_c = self.clfC.predict_proba(val[featureC])[:,1]  
            """ 
                分别计算三个模型在val上的AUC 
                模型加权融合的策略：以单模型的AUC作为权重
            """  
            valid_scoreA = AUC(val[bad_ind],valid_a)  
            valid_scoreB = AUC(val[bad_ind],valid_b)  
            valid_scoreC = AUC(val[bad_ind],valid_c)  
            valid_score = AUC(val[bad_ind], valid_a*valid_scoreA
                                             +valid_b*valid_scoreB + valid_c*valid_scoreC)
              
            """ 
                index1 三个模型在test上的预测概率相同的样本 
                sum_va 三个模型AUC之和为分母做归一化 
                prob 测试集分类结果融合， 
                index1（分类结果）*AUC（权重）/sum_va（归一化分母） 
                index2 分类结果升序排列，取出两端的test样本 
                new 筛选后样本集 
            """  
            index1 = (preA==preB) & (preA==preC)  
            sum_va = valid_scoreA+valid_scoreB+valid_scoreC  
            prob = (probC[index1]*valid_scoreC+probA[index1]*valid_scoreA  
                    +probB[index1]*valid_scoreB)/sum_va  
            Ap_low = np.sort(prob)[int(len(prob)*turn/2.0/self.max_turns)]-0.01  
            Ap_high= np.sort(prob)[int(len(prob)*
                                                          (1-turn/2.0/self.max_turns))]+0.01
            index2 = ((prob>Ap_high) | (prob<Ap_low))    
            new['no'] = test['no'][index1][index2]      
            new['pred'] = prob[index2]  
            result = result.append(new)  
            """ 
                rightSamples 同时满足index1和index2条件的预测概率 
                score_sim 三个模型在test上的预测结果差异和 
            """  
            rightSamples = test[index1][index2]  
            rightSamples[bad_ind] = preA[index1][index2]  
  
            score_sim = np.sum(abs(probA-probB)+
                                             abs(probA-probC) +abs(probB-probC)+0.1)/len(probA)
            """ 
                从数据集dev中取出step之后的部分样本并计算AUC 
                valid_score 前文三模型加权融合的AUC 
                得到drop 
            """  
            true_y = dev.iloc[self.step:][bad_ind]  
            dev_prob = predA[self.step:]*valid_scoreA+ predB[self.step:]*valid_scoreB + predC[self.step:]*valid_scoreC  
                              
            dev_score = AUC(true_y,dev_prob)  
              
            drop = self.max_turns/(1+ drop_rate*
                                                      np.exp(-self.max_turns)*valid_score)
            """ 
                使用Traddaboost相同的权重调整方法， 
                挑选权重大于阈值的样本。 
            """  
            loss_bias = 0  
            if(self.step>0):  
                true_y = dev.iloc[0:self.step][bad_ind]  
                temp = predA[0:self.step]*valid_scoreA  \  
                        + predB[0:self.step]*valid_scoreB  \  
                        + predC[0:self.step]*valid_scoreC  
                temp = (temp+0.1)/(max(temp)+0.2)#归一化  
                temp = (true_y-1)*np.log(1-temp)-true_y*np.log(temp)#样本权重  
                loc = int(min(self.step,len(rightSamples)*drop+2)
                                                             *np.random.rand())#去除样本的比例  
                loss_bias =  np.sort(temp)[-loc]  
                temp = np.append(temp,np.zeros(len(dev)-self.step)-99)  
                remain_index = (temp <= loss_bias)  
                self.step = self.step-sum(1-remain_index)  
            else:  
                remain_index = []  
                  
            """ 
                得到新的test 
            """  
            dev = dev[remain_index].append(rightSamples[features+[bad_ind,'no']])  
            test = test[~test.index.isin(rightSamples.index)]  
            turn += 1  
        """ 
            计算原始test上的AUC 
        """  
        probA = self.clfA.predict_proba(test[featureA])[:,1]  
        pA = self.clfA.predict_proba(temp_test[featureA])[:,1]  
        valid_a = self.clfA.predict_proba(val[featureA])[:,1]  
  
        probB = self.clfB.predict_proba(test[featureB])[:,1]  
        valid_b = self.clfB.predict_proba(val[featureB])[:,1]  
        pB = self.clfB.predict_proba(temp_test[featureB])[:,1]  
  
        probC = self.clfC.predict_proba(test[features])[:,1]  
        valid_c = self.clfC.predict_proba(val[features])[:,1]  
        pC = self.clfC.predict_proba(temp_test[features])[:,1]  
  
        self.scoreA = AUC(val[bad_ind],valid_a)  
        self.scoreB = AUC(val[bad_ind],valid_b)  
        self.scoreC = AUC(val[bad_ind],valid_c)  

        return pA,pB,pC  

