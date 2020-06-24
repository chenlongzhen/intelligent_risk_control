# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:07:41 2019

@author: zixing.mei
"""

import networkx as nx    
import numpy as np    
from sklearn.model_selection import train_test_split    
from sklearn.neighbors import KNeighborsClassifier    
from sklearn.svm import SVC    
#给定真实标签    
G = nx.karate_club_graph()    
groundTruth = [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,0,0,0,0,1,1]
#定义邻接矩阵，将网络节点转换成n*n的方阵   
def graphmatrix(G):    
    n = G.number_of_nodes()    
    temp = np.zeros([n,n])    
    for edge in G.edges():    
        temp[int(edge[0])][int(edge[1])] = 1     
        temp[int(edge[1])][int(edge[0])] = 1    
    return temp    
    
edgeMat = graphmatrix(G)    
    
x_train, x_test, y_train, y_test = train_test_split(edgeMat, 
                                     groundTruth, test_size=0.6, random_state=0)
#使用线性核svm分类器进行训练    
clf = SVC(kernel="linear")    
    
clf.fit(x_train, y_train)    
predicted= clf.predict(x_test)     
print(predicted)    
    
score = clf.score(x_test, y_test)    
print(score)   

import networkx as nx  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
#二值化，默认用0.5作为阈值，可以根据业务标签分布调整   
def binary(nodelist, threshold=0.5):  
    for i in range(len(nodelist)):  
        if( nodelist[i] > threshold ): nodelist[i] = 1.0  
        else: nodelist[i] = 0  
    return nodelist  
  
G = nx.karate_club_graph()  
groundTruth = [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,0,0,0,0,1,1]  
max_iter = 2 #迭代次数  
nodes = list(G.nodes())   
nodes_list = {nodes[i]: i for i in range(0, len(nodes))}  
  
vote = np.zeros(len(nodes))  
x_train, x_test, y_train, y_test = train_test_split(nodes, 
                                       groundTruth, test_size=0.7, random_state=1)
  
vote[x_train] = y_train  
vote[x_test] = 0.5 #初始化概率为0.5  
  
for i in range(max_iter):  
    #只用前一次迭代的值  
    last = np.copy(vote)  
    for u in G.nodes():
        if( u in x_train ):
            continue  
        temp = 0.0  
        for item in G.neighbors(u):  
            #对所有邻居求和  
            temp = temp + last[nodes_list[item]]  
        vote[nodes_list[u]] = temp/len(list(G.neighbors(u)))  
 
#二值化得到分类标签   
temp = binary(vote)  
  
pred = temp[x_test]  
#计算准确率   
print(accuracy_score(y_test, pred))  
import networkx as nx  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
from sklearn import preprocessing  
from scipy import sparse  
  
G = nx.karate_club_graph()  
groundTruth = [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,0,0,0,0,1,1]  
  
def graphmatrix(G):  
    #节点抽象成边  
    n = G.number_of_nodes()  
    temp = np.zeros([n,n])  
    for edge in G.edges():  
        temp[int(edge[0])][int(edge[1])] = 1   
        temp[int(edge[1])][int(edge[0])] = 1  
    return temp  
  
def propagation_matrix(G):  
    #矩阵标准化  
    degrees = G.sum(axis=0)  
    degrees[degrees==0] += 1  # 避免除以0  
      
    D2 = np.identity(G.shape[0])  
    for i in range(G.shape[0]):  
        D2[i,i] = np.sqrt(1.0/degrees[i])  
      
    S = D2.dot(G).dot(D2)  
    return S  
#定义取最大值的函数   
def vec2label(Y):  
    return np.argmax(Y,axis=1)  
  
edgematrix = graphmatrix(G)  
S = propagation_matrix(edgematrix)  
  
Ap = 0.8  
cn = 2  
max_iter = 10  
  
#定义迭代函数  
F = np.zeros([G.number_of_nodes(),2])  
X_train, X_test, y_train, y_test = train_test_split(list(G.nodes()), 
                                       groundTruth, test_size=0.7, random_state=1)
for (node, label) in zip(X_train, y_train):  
    F[node][label] = 1  
  
Y = F.copy()  
  
for i in range(max_iter):  
    F_old = np.copy(F)  
    F = Ap*np.dot(S, F_old) + (1-Ap)*Y  
  
temp = vec2label(F)  
pred = temp[X_test]  
print(accuracy_score(y_test, pred))  

