# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:11:33 2019

@author: zixing.mei
"""

import networkx as nx  
from networkx.algorithms import community  
import itertools  
  
G = nx.karate_club_graph()  
comp = community.girvan_newman(G)     
# 令社区个数为4，这样会依次得到K=2，K=3，K=4时候的划分结果
k = 4  
limited = itertools.takewhile(lambda c: len(c) <= k, comp)  
for communities in limited:  
    print(tuple(sorted(c) for c in communities)) 
    
    
import networkx as nx  
import community   
G = nx.karate_club_graph()  
part = community.best_partition(G)  
print(len(part)) 


import math  
import numpy as np  
from sklearn import metrics  
def NMI(A,B):  
    total = len(A)  
    X = set(A)  
    Y = set(B)  
    #计算互信息MI  
    MI = 0  
    eps = 1.4e-45  
    for x in X:  
        for y in Y:  
            AOC = np.where(A==x)  
            BOC = np.where(B==y)  
            ABOC = np.intersect1d(AOC,BOC)  
            px = 1.0*len(AOC[0])/total  
            py = 1.0*len(BOC[0])/total  
            pxy = 1.0*len(ABOC)/total  
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)  
    # 标准化互信息NMI  
    Hx = 0  
    for x in X:  
        AOC = 1.0*len(np.where(A==x)[0])  
        Hx = Hx - (AOC/total)*math.log(AOC/total+eps,2)  
    Hy = 0  
    for y in Y:  
        BOC = 1.0*len(np.where(B==y)[0])  
        Hy = Hy - (BOC/total)*math.log(BOC/total+eps,2)  
    NMI = 2.0*MI/(Hx+Hy)  
    return NMI  
#测试   
if __name__ == '__main__':  
    A = np.array([1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3])  
    B = np.array([1,2,1,1,1,1,1,2,2,2,2,3,1,1,3,3,3]) 
       #调用自定义的NMI函数 
    print(NMI(A,B))  
        #调用sklearn封装好的NMI函数
    print(metrics.normalized_mutual_info_score(A,B))

