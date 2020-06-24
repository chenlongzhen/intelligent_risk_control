# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:41:08 2019

@author: zixing.mei
"""

n_components = np.arange(1, 21)  
models = [GMM(n, covariance_type='full', 
                random_state=0).fit(Xmoon) for n in n_components]  
plt.plot(n_components, [m.bic(Xmoon) for m in models], label='BIC')  
plt.plot(n_components, [m.aic(Xmoon) for m in models], label='AIC')  
plt.legend(loc='best')  
plt.xlabel('n_components')  
