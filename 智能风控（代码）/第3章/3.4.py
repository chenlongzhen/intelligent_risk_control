# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:30:41 2019

@author: zixing.mei
"""

import numpy as np  
from scipy.linalg.misc import norm  
from scipy.sparse.linalg import eigs  
  
def JDA(Xs,Xt,Ys,Yt,k=100,lamda=0.1,ker='primal',gamma=1.0,data='default'):  
    X = np.hstack((Xs , Xt))  
    X = np.diag(1/np.sqrt(np.sum(X**2)))  
    (m,n) = X.shape  
    #源域样本量  
    ns = Xs.shape[1]  
    #目标域样本量  
    nt = Xt.shape[1]  
    #分类个数  
    C = len(np.unique(Ys))  
    # 生成MMD矩阵  
    e1 = 1/ns*np.ones((ns,1))  
    e2 = 1/nt*np.ones((nt,1))  
    e = np.vstack((e1,e2))  
    M = np.dot(e,e.T)*C  
      
    #除了0，空，False以外都可以运行  
    if any(Yt) and len(Yt)==nt:  
        for c in np.reshape(np.unique(Ys) ,-1 ,1):  
            e1 = np.zeros((ns,1))  
            e1[Ys == c] = 1/len(Ys[Ys == c])  
            e2 = np.zeros((nt,1))  
            e2[Yt ==c] = -1/len(Yt[Yt ==c])  
            e = np.hstack((e1 ,e2))  
            e = e[np.isinf(e) == 0]  
            M = M+np.dot(e,e.T)  
      
    #矩阵迹求平方根          
    M = M/norm(M ,ord = 'fro' )  
      
    # 计算中心矩阵  
    H = np.eye(n) - 1/(n)*np.ones((n,n))  
      
    # Joint Distribution Adaptation: JDA  
    if ker == 'primal':  
        #特征值特征向量  
        A = eigs(np.dot(np.dot(X,M),X.T)+lamda*np.eye(m),
                           k=k, M=np.dot(np.dot(X,H),X.T),  which='SM')  
        Z = np.dot(A.T,X)  
    else:  
        pass  
    return A,Z  

