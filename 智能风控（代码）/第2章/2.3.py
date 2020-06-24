# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:52:55 2019

@author: zixing.mei
"""

from sklearn.preprocessing import OneHotEncoder   
enc = OneHotEncoder()  
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])   
enc.transform([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]).toarray()  

import math  
#离散型变量 WOE编码  
class charWoe(object):  
    def __init__(self, datasets, dep, weight, vars):  
                #数据集字典，{'dev':训练集,'val':测试集,'off':跨时间验证集}  
        self.datasets = datasets 
        self.devf = datasets.get("dev", "") #训练集  
        self.valf = datasets.get("val", "") #测试集  
        self.offf = datasets.get("off", "") #跨时间验证集  
        self.dep = dep #标签  
        self.weight = weight #样本权重  
        self.vars = vars #参与建模的特征名  
        self.nrows, self.ncols = self.devf.shape #样本数，特征数  
  
    def char_woe(self):  
        #得到每一类样本的个数，且加入平滑项使得bad和good都不为0  
        dic = dict(self.devf.groupby([self.dep]).size())  
        good  = dic.get(0, 0) + 1e-10
        bad =  dic.get(1, 0) + 1e-10  
        #对每一个特征进行遍历。  
        for col in self.vars:  
            #得到每一个特征值对应的样本数。  
            data = dict(self.devf[[col, self.dep]].groupby(
                                                  [col, self.dep]).size())  
            ''' 
            当前特征取值超过100个的时候，跳过当前取值。 
            因为取值过多时，WOE分箱的效率较低，建议对特征进行截断。 
            出现频率过低的特征值统一赋值，放入同一箱内。 
            '''  
            if len(data) > 100:  
                print(col, "contains too many different values...")
                continue  
            #打印取值个数  
            print(col, len(data))  
            dic = dict()  
            #k是特征名和特征取值的组合，v是样本数  
            for (k, v) in data.items():  
                #value为特征名，dp为特征取值  
                value, dp = k  
                #如果找不到key设置为一个空字典  
                dic.setdefault(value, {})   
                #字典中嵌套字典  
                dic[value][int(dp)] = v  
            for (k, v) in dic.items():  
                dic[k] = {str(int(k1)): v1 for (k1, v1) in v.items()}  
                dic[k]["cnt"] = sum(v.values())  
                bad_rate = round(v.get("1", 0)/ dic[k]["cnt"], 5)  
                dic[k]["bad_rate"] = bad_rate  
            #利用定义的函数进行合并。  
            dic = self.combine_box_char(dic)  
            #对每个特征计算WOE值和IV值  
            for (k, v) in dic.items():  
                a = v.get("0", 1) / good + 1e-10  
                b = v.get("1", 1) / bad + 1e-10  
                dic[k]["Good"] = v.get("0", 0)  
                dic[k]["Bad"] = v.get("1", 0)  
                dic[k]["woe"] = round(math.log(a / b), 5)  
            ''' 
            按照分箱后的点进行分割， 
            计算得到每一个特征值的WOE值， 
            将原始特征名加上'_woe'后缀，并赋予WOE值。 
            '''  
            for (klis, v) in dic.items():  
                for k in klis.split(","):  
                    #训练集进行替换  
                    self.devf.loc[self.devf[col]==k,
                                                    "%s_woe" % col] = v["woe"]
                    #测试集进行替换  
                    if not isinstance(self.valf, str):  
                        self.valf.loc[self.valf[col]==k,
                                                     "%s_woe" % col] = v["woe"]
                    #跨时间验证集进行替换  
                    if not isinstance(self.offf, str):  
                        self.offf.loc[self.offf[col]==k,                     
                                                     "%s_woe" % col] = v["woe"]
        #返回新的字典，其中包含三个数据集。  
        return {"dev": self.devf, "val": self.valf, "off": self.offf}
  
    def combine_box_char(self, dic):  
        ''' 
        实施两种分箱策略。 
        1.不同箱之间负样本占比差异最大化。 
        2.每一箱的样本量不能过少。 
        '''  
        #首先合并至10箱以内。按照每一箱负样本占比差异最大化原则进行分箱。  
        while len(dic) >= 10:  
            #k是特征值，v["bad_rate"]是特征值对应的负样本占比
            bad_rate_dic = {k: v["bad_rate"] 
                                             for (k, v) in dic.items()}  
            #按照负样本占比排序。因为离散型变量 是无序的，
                        #可以直接写成负样本占比递增的形式。  
            bad_rate_sorted = sorted(bad_rate_dic.items(),
                                                         key=lambda x: x[1])
            #计算每两箱之间的负样本占比差值。
                        #准备将差值最小的两箱进行合并。  
            bad_rate = [bad_rate_sorted[i+1][1]-
                                      bad_rate_sorted[i][1] for i in 
                                      range(len(bad_rate_sorted)-1)]
            min_rate_index = bad_rate.index(min(bad_rate))  
            #k1和k2是差值最小的两箱的key.  
            k1, k2 = bad_rate_sorted[min_rate_index][0],\
                                     bad_rate_sorted[min_rate_index+1][0]  
            #得到重新划分后的字典，箱的个数比之前少一。  
            dic["%s,%s" % (k1, k2)] = dict()  
            dic["%s,%s" % (k1, k2)]["0"] = dic[k1].get("0", 0)\
                                                            + dic[k2].get("0", 0)
            dic["%s,%s" % (k1, k2)]["1"] = dic[k1].get("1", 0) \
                                                            + dic[k2].get("1", 0)
            dic["%s,%s" % (k1, k2)]["cnt"] = dic[k1]["cnt"]\
                                                              + dic[k2]["cnt"]
            dic["%s,%s" % (k1, k2)]["bad_rate"] = round(
                                    dic["%s,%s" % (k1, k2)]["1"] / 
                                    dic["%s,%s" % (k1, k2)]["cnt"],5)  
            #删除旧的key。  
            del dic[k1], dic[k2]  
        ''' 
        结束循环后，箱的个数应该少于10。 
        下面实施第二种分箱策略。 
        将样本数量少的箱合并至其他箱中，以保证每一箱的样本数量不要太少。 
        '''  
        #记录当前样本最少的箱的个数。      
        min_cnt = min([v["cnt"] for v in dic.values()])  
        #当样本数量小于总样本的5%或者总箱的个数大于5的时候，对箱进行合并  
        while min_cnt < self.nrows * 0.05 and len(dic) > 5:  
            min_key = [k for (k, v) in dic.items() 
                                     if v["cnt"] == min_cnt][0]  
            bad_rate_dic = {k: v["bad_rate"] 
                                          for (k, v) in dic.items()}  
            bad_rate_sorted = sorted(bad_rate_dic.items(),
                                              key=lambda x: x[1])  
            keys = [k[0] for k in bad_rate_sorted]  
            min_index = keys.index(min_key)  
            ''''' 
            同样想保持合并后箱之间的负样本占比差异最大化。 
            由于箱的位置不同，按照三种不同情况进行分类讨论。 
            '''  
            #如果是第一箱，和第二项合并  
            if min_index == 0:  
                k1, k2 = keys[:2]  
            #如果是最后一箱，和倒数第二箱合并  
            elif min_index == len(dic) - 1:  
                k1, k2 = keys[-2:]  
            #如果是中间箱，和bad_rate值相差最小的箱合并  
            else:  
                bef_bad_rate = dic[min_key]["bad_rate"]\
                                             -dic[keys[min_index - 1]]["bad_rate"]
                aft_bad_rate = dic[keys[min_index+1]]["bad_rate"] - dic[min_key]["bad_rate"]
                if bef_bad_rate < aft_bad_rate:  
                    k1, k2 = keys[min_index - 1], min_key
                else:  
                    k1, k2 = min_key, keys[min_index + 1]
            #得到重新划分后的字典，箱的个数比之前少一。  
            dic["%s,%s" % (k1, k2)] = dict()  
            dic["%s,%s" % (k1, k2)]["0"] = dic[k1].get("0", 0) \
                                                             + dic[k2].get("0", 0)
            dic["%s,%s" % (k1, k2)]["1"] = dic[k1].get("1", 0)\
                                                             + dic[k2].get("1", 0)
            dic["%s,%s" % (k1, k2)]["cnt"] = dic[k1]["cnt"]\
                                                                  +dic[k2]["cnt"]
            dic["%s,%s" % (k1, k2)]["bad_rate"] = round(
                                                dic["%s,%s" % (k1, k2)]["1"] / 
                                                dic["%s,%s" % (k1, k2)]["cnt"],5)
            #删除旧的key。  
            del dic[k1], dic[k2]  
            #当前最小的箱的样本个数  
            min_cnt = min([v["cnt"] for v in dic.values()])  
        return dic  

