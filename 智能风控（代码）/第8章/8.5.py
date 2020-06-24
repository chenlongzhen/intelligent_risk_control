# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:23:41 2019

@author: zixing.mei
"""

import numpy as np  
  
def get_cost(X, U, V, lamb=0):  
    '''''  
        计算损失函数     
        J = |X-UV|+ lamb*(|U|+|V|) 
        输入: X [n, d], U [n, m], V [m, d]  
    '''  
    UV = np.dot(U, V)   
    cost1 = np.sum((X - UV)**2)  
    cost2 = np.sum(U**2) + np.sum(V**2)  
    res = cost1 + lamb*cost2  
    return res  
  
def Matrix_Factor(X, m, lamb=0.1, learnRate=0.01):  
    '''''  
        损失函数定义 
        J = |X-UV| + lamb*(|U|+|V|) 
        输入: X [n, d]  
        输出: U [n, m], V [m, n] 
    '''  
    maxIter = 100  
    n, d = X.shape  
    #随机初始化  
    U = np.random.random([n, m])/n  
    V = np.random.random([m, d])/m  
    # 迭代  
    iter_num = 1   
    while iter_num < maxIter:  
        #计算U的偏导  
        dU = 2*( -np.dot(X, V.T) + np.linalg.multi_dot([U, V, V.T]) + lamb*U )
        U = U - learnRate * dU  
        #计算V的偏导  
        dV = 2*( -np.dot(U.T, X) + np.linalg.multi_dot([U.T, U, V]) + lamb*V )
        V = V - learnRate * dV  
        iter_num += 1  
    return U, V  


import numpy as np  
import networkx as nx  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import roc_curve,auc  
from matplotlib import pyplot as plt   
import random
#加载数据 
G = nx.karate_club_graph()  
groundTruth = [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1]  
  
#构造邻接矩阵  
def graph2matrix(G):  
    n = G.number_of_nodes()  
    res = np.zeros([n,n])  
    for edge in G.edges():  
        res[int(edge[0])][int(edge[1])] = 1   
        res[int(edge[1])][int(edge[0])] = 1  
    return res  
  
#生成网络  
G = nx.karate_club_graph()  
G = graph2matrix(G)  
  
#迭代20次  
[U, V] = Matrix_Factor(G, 20)  
#划分训练集测试集  
X_train, X_test, y_train, y_test = train_test_split(U,groundTruth,test_size=0.7,random_state=1)
#逻辑回归训练  
lgb_lm = LogisticRegression(penalty='l2',C=0.2,class_weight='balanced',solver='liblinear')  
lgb_lm.fit(X_train, y_train)   
   
y_pred_lgb_lm_train = lgb_lm.predict_proba(X_train)[:, 1]    
fpr_lgb_lm_train, tpr_lgb_lm_train, _ = roc_curve(y_train,y_pred_lgb_lm_train)  
  
y_pred_lgb_lm = lgb_lm.predict_proba(X_test)[:,1]    
fpr_lgb_lm,tpr_lgb_lm,_ = roc_curve(y_test,y_pred_lgb_lm)    
  
#计算KS值并绘制ROC曲线  
plt.figure(1)    
plt.plot([0, 1], [0, 1], 'k--')    
plt.plot(fpr_lgb_lm_train,tpr_lgb_lm_train,label='LGB + LR train')    
plt.plot(fpr_lgb_lm, tpr_lgb_lm, label='LGB + LR test')    
plt.xlabel('False positive rate')    
plt.ylabel('True positive rate')    
plt.title('ROC curve')    
plt.legend(loc='best')    
plt.show()    
print('train ks:',abs(fpr_lgb_lm_train - tpr_lgb_lm_train).max(),  
                'test AUC:',auc(fpr_lgb_lm_train, tpr_lgb_lm_train))  
print('test ks:',abs(fpr_lgb_lm - tpr_lgb_lm).max(),  
               ' test AUC:', auc(fpr_lgb_lm, tpr_lgb_lm)) 

def rondom_walk (self,length, start_node):  
    walk = [start_node]  
    while len(walk) < length:  
        temp = walk[-1]  
        temp_nbrs = list(self.G.neighbors(temp))  
        if len(temp_nbrs) > 0:  
            walk.append(random.choice(temp_nbrs))  
        else:  
            break  
    return walk  

#Node2Vec
import networkx as nx
from node2vec import Node2Vec
 
# 自定义图
graph = nx.fast_gnp_random_graph(n=100, p=0.5)
 
# 预计算概率并生成行走
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)  
 
# 嵌入节点
model = node2vec.fit(window=10, min_count=1, batch_words=4)  
 
# 寻找最相似节点
model.wv.most_similar('2')
 
# 保存节点嵌入结果
model.wv.save_word2vec_format('EMBEDDING_FILENAME')
 
# 保存模型
model.save('EMBEDDING_MODEL_FILENAME')
 
# 用Hadamard方法嵌入边
from node2vec.edges import HadamardEmbedder
 
edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
 
# 快速查找嵌入
edges_embs[('1', '2')]

# 在单独的实例中获取所有边
edges_kv = edges_embs.as_keyed_vectors()
 
# 寻找最相似边
edges_kv.most_similar(str(('1', '2')))
 
# 保存边嵌入结果
edges_kv.save_word2vec_format('EDGES_EMBEDDING_FILENAME')



