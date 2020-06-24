# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:50:38 2019

@author: zixing.mei
"""

def Num(feature,mth):  
    df=data.loc[:,feature +'1': feature +str(mth)]  
    auto_value=np.where(df>0,1,0).sum(axis=1)  
    return feature +'_num'+str(mth),auto_value 
 
def Avg(feature, mth):  
    df=data.loc[:,feature +'1': feature +str(mth)]  
    auto_value=np.nanmean(df,axis = 1 )  
    return feature +'_avg'+str(mth),auto_value  
   
def Msg(feature, mth):  
    df=data.loc[:,feature +'1': feature +str(mth)]  
    df_value=np.where(df>0,1,0)  
    auto_value=[]  
    for i in range(len(df_value)):  
        row_value=df_value[i,:]  
        if row_value.max()<=0:  
            indexs='0'  
            auto_value.append(indexs)  
        else:  
            indexs=1  
            for j in row_value:  
                if j>0:  
                    break  
                indexs+=1  
            auto_value.append(indexs)  
    return feature +'_msg'+str(mth),auto_value 

def Cav(feature, mth):  
    df=data.loc[:,feature +'1':inv+str(mth)]  
    auto_value = df[feature +'1']/np.nanmean(df,axis = 1 )   
    return feature +'_cav'+str(mth),auto_value 

def Mai(feature, mth):  
    arr=np.array(data.loc[:,feature +'1': feature +str(mth)])       
    auto_value = []  
    for i in range(len(arr)):  
        df_value = arr[i,:]  
        value_lst = []  
        for k in range(len(df_value)-1):  
            minus = df_value[k] - df_value[k+1]  
            value_lst.append(minus)  
        auto_value.append(np.nanmax(value_lst))       
    return feature +'_mai'+str(mth),auto_value 

def Ran(feature, mth):  
    df=data.loc[:,feature +'1': feature +str(mth)]  
    auto_value = np.nanmax(df,axis = 1 )  -  np.nanmin(df,axis = 1 )
    return feature +'_ran'+str(mth),auto_value   

