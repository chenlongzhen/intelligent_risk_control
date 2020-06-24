# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:42:49 2019

@author: zixing.mei
"""

class imbalanceData():    
      
    """  
      处理不均衡数据 
        train训练集 
        test测试集 
        mmin低分段错分比例 
        mmax高分段错分比例 
        bad_ind样本标签 
        lis不参与建模变量列表 
    """  
    def __init__(self, train,test,mmin,mmax, bad_ind,lis=[]):  
        self.bad_ind = bad_ind  
        self.train_x = train.drop([bad_ind]+lis,axis=1)  
        self.train_y = train[bad_ind]  
        self.test_x = test.drop([bad_ind]+lis,axis=1)  
        self.test_y = test[bad_ind]  
        self.columns = list(self.train_x.columns)  
        self.keep = self.columns + [self.bad_ind]  
        self.mmin = 0.1  
        self.mmax = 0.7  
      
    ''''' 
        设置不同比例， 
        针对头部和尾部预测不准的样本，进行加权处理。 
        0.1为噪声的权重，不参与过采样。 
        1为正常样本权重，参与过采样。 
    '''  
    def weight(self,x,y):  
        if x == 0 and y < self.mmin:  
            return 0.1  
        elif x == 1 and y > self.mmax:  
            return 0.1  
        else:  
            return 1  
    ''''' 
        用一个LightGBM算法和weight函数进行样本选择 
        只取预测准确的部分进行后续的smote过采样 
    '''  
    def data_cleaning(self):  
        lgb_model,lgb_auc  = self.lgb_test()  
        sample = self.train_x.copy()  
        sample[self.bad_ind] = self.train_y  
        sample['pred'] = lgb_model.predict_proba(self.train_x)[:,1]  
        sample = sample.sort_values(by=['pred'],ascending=False).reset_index()  
        sample['rank'] = np.array(sample.index)/len(sample)  
        sample['weight'] = sample.apply(lambda x:self.weight(x.bad_ind,x['rank']),
                                                                        axis = 1)
        osvp_sample = sample[sample.weight == 1][self.keep]  
        osnu_sample = sample[sample.weight < 1][self.keep]     
        train_x_osvp = osvp_sample[self.columns]  
        train_y_osvp = osvp_sample[self.bad_ind]  
        return train_x_osvp,train_y_osvp,osnu_sample  
  
    ''''' 
        实施smote过采样 
    '''  
    def apply_smote(self):  
        ''''' 
            选择样本，只对部分样本做过采样 
            train_x_osvp,train_y_osvp 为参与过采样的样本 
            osnu_sample为不参加过采样的部分样本 
        '''  
        train_x_osvp,train_y_osvp,osnu_sample = self.data_cleaning()  
        rex,rey = self.smote(train_x_osvp,train_y_osvp)  
        print('badpctn:',rey.sum()/len(rey))  
        df_rex = pd.DataFrame(rex)  
        df_rex.columns =self.columns  
        df_rex['weight'] = 1  
        df_rex[self.bad_ind] = rey  
        df_aff_ovsp = df_rex.append(osnu_sample)  
        return df_aff_ovsp  
  
    ''''' 
        定义LightGBM函数 
    '''  
    def lgb_test(self):  
        import lightgbm as lgb  
        clf =lgb.LGBMClassifier(boosting_type = 'gbdt',  
                               objective = 'binary',  
                               metric = 'auc',  
                               learning_rate = 0.1,  
                               n_estimators = 24,  
                               max_depth = 4,  
                               num_leaves = 25,  
                               max_bin = 40,  
                               min_data_in_leaf = 5,  
                               bagging_fraction = 0.6,  
                               bagging_freq = 0,  
                               feature_fraction = 0.8,  
                               )  
        clf.fit(self.train_x,self.train_y,eval_set=[(self.train_x,self.train_y),
                                                                  (self.test_x,self.test_y)],
                                                                    eval_metric = 'auc')
        return clf,clf.best_score_['valid_1']['auc']  
  
    ''''' 
        调用imblearn中的smote函数 
    '''  
    def smote(self,train_x_osvp,train_y_osvp,m=4,K=15,random_state=0):  
        from imblearn.over_sampling import SMOTE  
        smote = SMOTE(k_neighbors=K, kind='borderline1', m_neighbors=m, n_jobs=1,  
                out_step='deprecated', random_state=random_state, ratio=None,  
                      svm_estimator='deprecated')  
        rex,rey = smote.fit_resample(train_x_osvp,train_y_osvp)  
        return rex,rey  
df_aff_ovsp = imbalanceData(train=train,test=evl,mmin=0.3,mmax=0.7, bad_ind='bad_ind',
                            lis=['index', 'uid', 'td_score', 'jxl_score', 'mj_score',
                                 'rh_score', 'zzc_score', 'zcx_score','obs_mth']).apply_smote()
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import roc_curve  
  
lr_model = LogisticRegression(C=0.05,class_weight='balanced')  
lr_model.fit(x,y)  
  
y_pred = lr_model.predict_proba(x)[:,1]  
fpr_lr_train,tpr_lr_train,_ = roc_curve(y,y_pred)  
train_ks = abs(fpr_lr_train - tpr_lr_train).max()  
print('train_ks : ',train_ks)  
  
y_pred = lr_model.predict_proba(evl_x)[:,1]  
fpr_lr,tpr_lr,_ = roc_curve(evl_y,y_pred)  
evl_ks = abs(fpr_lr - tpr_lr).max()  
print('evl_ks : ',evl_ks)  
  
from matplotlib import pyplot as plt  
plt.plot(fpr_lr_train,tpr_lr_train,label = 'train LR')  
plt.plot(fpr_lr,tpr_lr,label = 'evl LR')  
plt.plot([0,1],[0,1],'k--')  
plt.xlabel('False positive rate')  
plt.ylabel('True positive rate')  
plt.title('ROC Curve')  
plt.legend(loc = 'best')  
plt.show() 

