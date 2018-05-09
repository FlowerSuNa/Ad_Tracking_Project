
## 2. Train Data Preprocessing
## Import library
import pandas as pd
import numpy as np
import gc


## Extract a sample
import random

ad = pd.read_csv("train.csv")
gc.collect()

print(ad.shape)
print(ad.columns)

for n in [10000000, 20000000,30000000]:
    idx = random.sample(range(len(ad)),n)
    sample = ad.iloc[idx]
    gc.collect()
    
    del idx
    gc.collect()

    n = n / 1000000
    sample.to_csv('train_' + str(n) + 'm.csv', index=False)
    
    del sample
    gc.collect()
    
del ad
gc.collect


## Import data
# ad = pd.read_csv("train_10m.csv", parse_dates=['click_time'])
# ad = pd.read_csv("train_20m.csv", parse_dates=['click_time'])
# ad = pd.read_csv("train_30m.csv", parse_dates=['click_time'])
ad = pd.read_csv("train.csv", parse_dates=['click_time'])
gc.collect()

print(ad.shape)
print(ad.columns)


## Make a derived variable : hour
ad['hour'] = np.nan
ad['hour'] = ad['click_time'].dt.hour
gc.collect()

print(ad[['click_time','hour']].head(10))
print(ad[['click_time','hour']].tail(10))


## Remove variables
del ad['click_time']
del ad['attributed_time']
gc.collect()


## Make derived variables
## 'v'_cnt : the frequency of 'v'
## 'v'_attr : the number of download by 'v'
## 'v'_attr_prop : the proporation of download by 'v'
## tot_attr_prop : the total of 'v'_attr_prop
var = ['ip','app','device','os','channel','hour']
var1 = ['ip_cnt','app_cnt','device_cnt','os_cnt','channel_cnt','hour_cnt']
var2 = ['ip_attr','app_attr','device_attr','os_attr','channel_attr','hour_attr']
var3 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop','channel_attr_prop','hour_attr_prop']

for v,v1,v2,v3 in zip(var,var1,var2,var3):
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
    
    del ad[v1]
    del ad[v2]
    gc.collect()    
    
    print(ad[[v,v3]].head(10))
    print(ad[[v,v3]].tail(10))

ad['tot_attr_prop'] = np.nan
ad['tot_attr_prop'] = ad[var3].sum(axis=1)
gc.collect()

print(ad['tot_attr_prop'].head(10))
print(ad['tot_attr_prop'].tail(10))


## 'v'_'vv'_cnt : frequency by 'v' and 'vv'
## 'v'_'vv'_attr : the number of download by 'v' and 'vv'
## 'v'_'vv'_prop : the proporation of download by 'v' and 'vv'
## tot_vv_prop : The total of 'v'_'vv'_prop
var4 = ['ip_hour_prop','ip_app_prop','ip_channel_prop','hour_app_prop','hour_channel_prop']

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
        ad[prop] = (ad[attr] / ad[cnt])
        gc.collect()    
        
        del ad[cnt]
        del ad[attr]
        gc.collect()
        
        print(ad[[v,vv,prop]].head(10))
        print(ad[[v,vv,prop]].tail(10))
        
ad['tot_vv_prop'] = np.nan
ad['tot_vv_prop'] = ad[var4].sum(axis=1)
gc.collect()

print(ad['tot_vv_prop'].head(10))
print(ad['tot_vv_prop'].tail(10)) 
        

## Check correlation
feat = var3 + var4 + ['is_attributed']
print(ad[feat].corr(method='pearson'))

pd.plotting.scatter_matrix(ad[var3 + ['is_attributed']], figsize=(15,15), alpha=.1, diagonal='kde')
pd.plotting.scatter_matrix(ad[var4 + ['is_attributed']], figsize=(15,15), alpha=.1, diagonal='kde')


## Save dataset
# ad.to_csv('train_10m_modify1.csv', index=False)
# ad.to_csv('train_20m_modify1.csv', index=False)
# ad.to_csv('train_30m_modify1.csv', index=False)
ad.to_csv('train_modify1.csv', index=False)