# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:55:52 2019

@author: zixing.mei
"""

from sklearn.metrics import accuracy_score  
import lightgbm as lgb 

#'regression_l1'等价于MAE损失函数
lgb_param_l1 = {  
    'learning_rate': 0.01,  
    'boosting_type': 'gbdt',  
    'objective': 'regression_l1',   
    'min_child_samples': 46,  
    'min_child_weight': 0.02,  
    'feature_fraction': 0.6,  
    'bagging_fraction': 0.8,  
    'bagging_freq': 2,  
    'num_leaves': 31,  
    'max_depth': 5,  
    'lambda_l2': 1,  
    'lambda_l1': 0,  
    'n_jobs': -1,  
}  
  
#'regression_l2'等价于MSE损失函数  
lgb_param_l2 = { 
    'learning_rate': 0.01,  
    'boosting_type': 'gbdt',  
    'objective': 'regression_l2',  
    'feature_fraction': 0.7,  
    'bagging_fraction': 0.7,  
    'bagging_freq': 2,  
    'num_leaves': 52,  
    'max_depth': 5,  
    'lambda_l2': 1,  
    'lambda_l1': 0,  
    'n_jobs': -1,  
}  
# 第一种参数预测  
clf1=lgb.LGBMRegressor(**lgb_params1)  
clf.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_val,y_val)],
                                  eval_metric='mae',early_stopping_rounds=200)
#预测的划分出来的测试集的标签  
pred_val1=clf1.predict(X_val,num_iteration=clf.best_iteration_)   
vali_mae1=accuracy_score(y_val,np.round(pred_val1))  
#预测的未带标签的测试集的标签  
pred_test1=clf.predcit(test[feature_name],num_iteration=clf.best_iteration_)   
# 第二种参数预测 
clf2=lgb.LGBMRegressor(**lgb_params2)
clf2.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_val,y_val)],
                                   eval_metric='rmse',early_stopping_rounds=200)  
#预测的划分出来的测试集的标签  
pred_val2=clf2.predict(X_val,num_iteration=clf2.best_iteration_)  
vali_mae2=accuracy_score(y_val,np.round(pred_val2))  
#预测的未带标签的测试集的标签  
pred_test2=clf.predcit(test_featur,num_iteration=clf2.best_iteration_)   
# 模型参数进行融合之后的结果  
pred_test=pd.DataFrame()  
pred_test['ranks']=list(range(50000))  
pred_test['result']=1  
pred_test.loc[pred_test.ranks<400,'result'] = 
           pred_test1.loc[pred_test1.ranks< 400,'pred_mae'].values *0.4 
           + pred_test2.loc[pred_test2.ranks< 400,'pred_mse'].values * 0.6  
pred_test.loc[pred_test.ranks>46000,'result'] = 
              pred_test1.loc[pred_test1.ranks> 46000,'pred_mae'].values *0.4
              + pred_test2.loc[pred_test2.ranks> 46000,'pred_mse'].values * 0.6

