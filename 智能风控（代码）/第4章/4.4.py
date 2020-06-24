# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:35:39 2019

@author: zixing.mei
"""

import matplotlib.pyplot as plt  
import seaborn as sns; sns.set()  
import numpy as np  
#产生实验数据  
from sklearn.datasets.samples_generator import make_blobs  
X, y_true = make_blobs(n_samples=700, centers=4,  
             cluster_std=0.5, random_state=2019) 
X = X[:, ::-1] #方便画图  
  
from sklearn.mixture import GaussianMixture as GMM  
gmm = GMM(n_components=4).fit(X) #指定聚类中心个数为4  
labels = gmm.predict(X)  
plt.scatter(X[:, 0], X[:, 1], c=labels, s=5, cmap='viridis')  
probs = gmm.predict_proba(X)  
print(probs[:10].round(2))  
size = probs.max(1)  
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size)  

from matplotlib.patches import Ellipse  
#给定的位置和协方差画一个椭圆
def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    #将协方差转换为主轴
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    #画出椭圆
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
#画图
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=4, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=4, zorder=2)
    ax.axis('equal')
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_  , gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
 
from sklearn.datasets import make_moons  
Xmoon, ymoon = make_moons(100, noise=.04, random_state=0)  
plt.scatter(Xmoon[:, 0], Xmoon[:, 1]) 
gmm2 = GMM(n_components=2, covariance_type='full', random_state=0)  
plot_gmm(gmm2, Xmoon) 
gmm10 = GMM(n_components=10, covariance_type='full', random_state=0)  
plot_gmm(gmm10, Xmoon, label=False)  
Xnew = gmm10.sample(200)[0]
plt.scatter(Xnew[:, 0], Xnew[:, 1])  

