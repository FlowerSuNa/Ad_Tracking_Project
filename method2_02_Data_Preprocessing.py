
## 2. Train Data Preprocessing
## Import library and data
import pandas as pd
import numpy as np
import gc

ad_train = pd.read_csv("train.csv", parse_dates=['click_time'])
print(ad_train.shape)
print(ad_train.columns)

ad_test = pd.read_csv("test.csv", parse_dates=['click_time'])
print(ad_test.shape)
print(ad_test.columns)


## Remove variables
del ad_train['attributed_time'] 
del ad_test['click_id']
gc.collect()


## Make and fill variable 'is_attributed' in test data
train_len = len(ad_train)
ad_test['is_attributed'] = ad_train.is_attributed.sum() / train_len
gc.collect()

print(ad_test['is_attributed'].head(10))
print(ad_test['is_attributed'].tail(10))


## Merge train data and test data
ad = pd.concat([ad_train, ad_test])
print(ad.shape)
gc.collect()


## Remove datasets
del ad_train
del ad_test
gc.collect()


## Make a derived variable 'hour'
ad['hour'] = np.nan
ad['hour'] = ad['click_time'].dt.hour
gc.collect()

print(ad[['click_time','hour']].head(10))
print(ad[['click_time','hour']].tail(10))


## Remove a variable
del ad['click_time']
gc.collect()


## Make derived variables
## 'v'_attr_prop : download proportion by 'v'
## 'v'_attr_tot_prop : download proportion among download by 'v'
## tot_attr_prop : the total of 'v'_attr_prop
## tot_attr_tot_prop : the total of 'v'_attr_tot_prop

var = ['ip','app','device','os','channel','hour']
var1 = ['ip_cnt','app_cnt','device_cnt','os_cnt','channel_cnt','hour_cnt']
var2 = ['ip_attr','app_attr','device_attr','os_attr','channel_attr','hour_attr']
var3 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop','channel_attr_prop','hour_attr_prop']
var4 = ['ip_attr_tot_prop','app_attr_tot_prop','device_attr_tot_prop','os_attr_tot_prop','channel_attr_tot_prop','hour_attr_tot_prop']

attr_sum = ad['is_attributed'].sum()
print(attr_sum)

for v,v1,v2,v3,v4 in zip(var,var1,var2,var3,var4):
    temp = ad[v].value_counts().reset_index(name='counts')
    temp.columns = [v,v1]
    ad = ad.merge(temp, on=v, how='left')
    gc.collect()

    temp = ad.groupby(v)['is_attributed'].sum().reset_index(name='counts')
    temp.columns = [v,v2]
    ad = ad.merge(temp, on=v, how='left')
    gc.collect()    

    ad[v3] = np.nan
    ad[v3] = ad[v2] / ad[v1]
    gc.collect()
    
    ad[v4] = np.nan
    ad[v4] = ad[v2] / attr_sum
    gc.collect()

    ## Remove variables
    del ad[v1]
    del ad[v2]
    
    print(ad[[v,v3,v4]].head(10))
    print(ad[[v,v3,v4]].tail(10))
    gc.collect()

ad['tot_attr_prop'] = np.nan
ad['tot_attr_prop'] = ad[var3].sum(axis=1)
gc.collect()

print(ad['tot_attr_prop'].head(10))
print(ad['tot_attr_prop'].tail(10))

ad['tot_attr_tot_prop'] = np.nan
ad['tot_attr_tot_prop'] = ad[var4].sum(axis=1)
gc.collect()

print(ad['tot_attr_tot_prop'].head(10))
print(ad['tot_attr_tot_prop'].tail(10))


## 'v'_'vv'_prop : download proportion by 'v' and 'vv'
## tot_vv_prop : the total of 'v'_'vv'_prop

var5 = ['ip_hour_prop','ip_app_prop','ip_channel_prop','hour_app_prop','hour_channel_prop']

for v in ['ip','hour']:
    if v == 'hour':
        v1 = ['app','channel']
    else:
        v1 = ['hour','app','channel']
    
    for vv in v1:
        cnt = v+'_'+vv+'_cnt'
        attr =  v+'_'+vv+'_attr'
        prop = v+'_'+vv+'_prop'
        
        temp = ad.groupby([v,vv])['is_attributed'].count().reset_index(name='counts')
        temp.columns = [v,vv,cnt]
        ad = ad.merge(temp, on=[v,vv], how='left')
        gc.collect()
        
        temp = ad.groupby([v,vv])['is_attributed'].sum().reset_index(name='counts')
        temp.columns = [v,vv,attr]
        ad = ad.merge(temp, on=[v,vv], how='left')
        gc.collect()
        
        ad[prop]= np.nan
        ad[prop] = ad[attr] / ad[cnt]
        gc.collect()
        
        ## Remove variables
        del ad[cnt]
        del ad[attr]
        gc.collect()       
                
        print(ad[[v,vv,prop]].head(10))
        print(ad[[v,vv,prop]].tail(10))
             
ad['tot_vv_prop'] = np.nan
ad['tot_vv_prop'] = ad[var5].sum(axis=1)
gc.collect()

print(ad['tot_vv_prop'].head(10)) 
print(ad['tot_vv_prop'].tail(10))      
 

## Check correlation
feat = var3 + ['tot_attr_prop'] + var4 + ['tot_attr_tot_prop'] + var5 + ['tot_vv_prop','is_attributed']

print(ad[feat].corr(method='pearson'))

pd.plotting.scatter_matrix(ad[var3 + ['tot_attr_prop','is_attributed']], figsize=(15,15), alpha=.1, diagonal='kde')
pd.plotting.scatter_matrix(ad[var4 + ['tot_attr_tot_prop','is_attributed']], figsize=(15,15), alpha=.1, diagonal='kde')
pd.plotting.scatter_matrix(ad[var5 + ['tot_vv_prop','is_attributed']], figsize=(15,15), alpha=.1, diagonal='kde')
       

## Save dataset
ad_train = ad.iloc[:train_len,]
ad_test = ad.iloc[train_len:,]
del ad
gc.collect()

ad_train.to_csv('train_modify2.csv', index=False)
ad_test.to_csv('test_modify2.csv', index=False)
del ad_test
gc.collect()


## Extract a sample
import random

for n in [10000000, 20000000,30000000,40000000,50000000]:
    idx = random.sample(range(len(ad_train)),n)
    sample = ad.iloc[idx]
    gc.collect()

    del idx
    gc.collect()

    n = n / 1000000
    sample.to_csv('train_' + str(n) + 'm_modify2.csv', index=False)
    
    del sample
