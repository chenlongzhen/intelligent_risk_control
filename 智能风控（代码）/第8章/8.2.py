# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:05:34 2019

@author: zixing.mei
"""

import networkx as nx  
import pandas as pd  
import matplotlib.pyplot as plt  
  
edge_list=pd.read_csv('./data/stack_network_links.csv')  
G=nx.from_pandas_edgelist(edge_list,edge_attr='value' )  
plt.figure(figsize=(15,10))  
nx.draw(G,with_labels=True,  
        edge_color='grey',  
        node_color='pink',  
        node_size = 500,  
        font_size = 40,  
        pos=nx.spring_layout(G,k=0.2))  
#度  
nx.degree(G) 
 
import networkx as nx  
nx.eigenvector_centrality(G)  

import networkx as nx  
nx.pagerank(G,Ap=0.9)  

import networkx as nx  
nx.betweenness_centrality(G)  

import networkx as nx  
nx.closeness_centrality(G)  

preds = nx.jaccard_coefficient(G, [('azure','.net')])  
for u, v, p in preds:  
    print('(%s, %s) -> %.8f' % (u, v, p)) 

