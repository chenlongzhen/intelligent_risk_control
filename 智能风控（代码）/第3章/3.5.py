# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:31:33 2019

@author: zixing.mei
"""

import numpy as np  
from scipy import sparse as sp  
def DAELM(Train_s,Train_t,Test_t,NL,Type="CLASSIFIER" , Num_hid=100 ,Active_Function="sig"):  
    ''' 
    Train_s：源域训练集
    Train_t：目标域训练集
    Test_t：目标域测试集
    Type：模型类型（分类："CLASSIFIER"，回归："REGRESSION"）  
    Num_hid：隐层神经元个数，默认100个 
    Active_Function：映射函数（" sigmoid ":sigmoid函数, "sin":正弦函数）
    NL：模型选择  
    '''  
      
    Cs = 0.01  
    Ct = 0.01  
      
    #回归或分类  
    REGRESSION=0  
    CLASSIFIER=1  
      
    #训练数据  
    train_data = Train_s  
    T = train_data[:,0].T  
    P = train_data[:,1:train_data.shape[1]].T  
    del train_data  
      
    #目标域数据  
    train_target_dt = Train_t  
    Tt = train_target_dt[:,0].T  
    Pt = train_target_dt[:,1:train_target_dt.shape[1]].T  
      
    #测试集数据  
    test_data = Test_t  
    TVT = test_data[:,0].T  
    TE0 = test_data[:,0].T  
    TVP = test_data[:,2:test_data.shape[1]].T  
    del test_data  
      
    Num_train = P.shape[1]  
    Num_train_Target = Pt.shape[1]  
    Num_test = TVP.shape[1]  
    Num_input= P.shape[0]  
      
    if Type is not "REGRESSION":  
        sorted_target = np.sort(np.hstack((T ,  TVT)))  
        label = np.zeros((1,1))  
        label[0,0] = sorted_target[0,0]  
        j = 0  
        for i in range(2,(Num_train+Num_test+1)):  
            if sorted_target[0,i-1] != label[0,j-1]:  
                j=j+1  
                label[0,j-1] = sorted_target[0,i-1]  
                  
        number_class = j+1  
        Num_output = number_class  
          
  
        temp_T = np.zeros(Num_output , Num_train)  
        for i in range(1,Num_train+1):  
            for j in range(1,number_class+1):  
                if label(0,j-1) == T(0,i-1):  
                    break  
            temp_T[j-1 , i-1] = 1  
        T = temp_T*2-1  
  
        Tt_m = np.zeros(Num_output , Num_train_Target)  
        for i in range(1,Num_train_Target+1):  
            for j in range(1 , number_class+1):  
                if label[0,j-1] == Tt[0,i-1]:  
                    break  
            Tt_m[j-1 , i-1] = 1  
        Tt = Tt_m*2-1  
          
  
        temp_TV_T = np.zeros(Num_output,Num_test)  
        for i in range(1,Num_test):  
            for j in range(1,number_class+1):  
                if label(0,j-1) == TVT(0,i-1):  
                    break  
            temp_TV_T[j-1 , i-1] = 1  
        TVT = temp_TV_T*2-1  
          
    InputWeight = np.random.rand(Num_hid,Num_input)*2-1  
    Bis_hid = np.random.rand(Num_hid ,1)  
    H_m = InputWeight*P  
    Ht_m = InputWeight*Pt  
    del P  
    del Pt  
      
    ind = np.ones(1,Num_train)  
    indt = np.ones(1,Num_train_Target)  
    BiasMatrix = Bis_hid[:,ind-1]  
    BiasMatrixT = Bis_hid[:,indt-1]  
    H_m = H_m + BiasMatrix  
    Ht_m=Ht_m+BiasMatrixT  
      
    if Active_Function == "sigmoid":  
        H = 1/(1+np.exp(-H_m))  
        Ht = 1/(1+np.exp(-Ht_m))  
    if Active_Function == "sin":  
        H = np.sin(H_m)  
        Ht = np.sin(Ht_m)  
    if Active_Function != " sigmoid " and Active_Function!="sin":  
        pass  
      
    del H_m  
    del Ht_m  
      
    n = Num_hid  
      
    #DAELM模型  
    H=H.T  
    Ht=Ht.T  
    T=T.T  
    Tt=Tt.T  
      
    if NL == 0:  
        A = Ht*H.T  
        B = Ht*Ht.T+np.eye(Num_train_Target)/Ct  
        C=H*Ht.T  
        D=H*H.T+np.eye(Num_train)/Cs  
        ApT=np.linalg.inv(B)*Tt-np.linalg.inv(B)*A* \
                       np.linalg.inv(C*np.linalg.inv(B)*A-D)*(C*np.linalg.inv(B)*Tt-T)
        ApS=inv(C*np.linalg.inv(B)*A-D)*(C*np.linalg.inv(B)*Tt-T)  
        OutputWeight=H.T*ApS+Ht.T*ApT  
    else:  
        OutputWeight=np.linalg.inv(np.eye(n)+Cs*H.t*H+Ct*Ht.T*Ht)*(Cs*H.T*T+Ct*Ht.T*Tt)  
      
    #计算准确率  
      
    Y=(H * OutputWeight).T  
      
    H_m_test=InputWeight*TVP  
    ind = np.ones(1,Num_hid)  
    BiasMatrix=Bis_hid[:,ind-1]  
    H_m_test = H_m_test+BiasMatrix  
    if Active_Function == "sig":  
        H_test = 1/(1+np.exp(-H_m_test))  
    if Active_Function == "sin":  
        H_test = np.sin(H_m_test)  
          
    TY = (H_test.T*OutputWeight).T  
      
    #返回测试集结果  
    if Type =="CLASSIFIER":  
        return TY  
    else:  
        pass  

