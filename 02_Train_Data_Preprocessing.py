
## 2. Train Data Preprocessing
## Import library and data
import pandas as pd
import numpy as np

ad = pd.read_csv("train_sample.csv", parse_dates=['click_time'])
print(ad.shape)
print(ad.columns)


## Make derived variables : hour, time
ad['hour'] = np.nan
ad['hour'] = ad['click_time'].dt.hour

ad['time'] = np.nan
ad['time'] = ad['hour'] // 4

print(ad[['click_time','hour','time']].head(20))


## Remove variables
del ad['click_time']
del ad['attributed_time']


## Make derived variables
## 'v'_cnt : frequency by 'v'
## 'v'_attr : the number of download by 'v'
## 'v'_attr_prop : the proporation of download by 'v'
## tot_attr_prop : the total of 'v'_attr_prop
var = ['ip','app','device','os','channel','hour','time']
var1 = ['ip_cnt','app_cnt','device_cnt','os_cnt','channel_cnt','hour_cnt','time_cnt']
var2 = ['ip_attr','app_attr','device_attr','os_attr','channel_attr','hour_attr','time_attr']
var3 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop',
        'channel_attr_prop','hour_attr_prop','time_attr_prop']

for v,v1,v2,v3 in zip(var,var1,var2,var3):
    temp = ad[v].value_counts().reset_index(name='counts')
    temp.columns = [v,v1]
    ad = ad.merge(temp, on=v, how='left')

    temp = ad.groupby(v)['is_attributed'].sum().reset_index(name='counts')
    temp.columns = [v,v2]
    ad = ad.merge(temp, on=v, how='left')

    ad[v3] = np.nan
    ad[v3] = ad[v2] / ad[v1]
    
    print(ad[[v,v1,v2,v3]].head(20))

ad['tot_attr_prop'] = np.nan
ad['tot_attr_prop'] = ad[var3].sum(axis=1)
print(ad['tot_attr_prop'].head(20))

## 'v'_'vv'_cnt : frequency by 'v' and 'vv'
## 'v'_'vv'_attr : the number of download by 'v' and 'vv'
## 'v'_'vv'_prop : the proporation of download by 'v' and 'vv'
## tot_vv_prop : The total of 'v'_'vv'_prop
var4 = ['ip_time_cnt','ip_app_cnt','ip_channel_cnt','time_app_cnt','time_channel_cnt']
var5 = ['ip_time_attr','ip_app_attr','ip_channel_attr','time_app_attr','time_channel_attr']
var6 = ['ip_time_prop','ip_app_prop','ip_channel_prop','time_app_prop','time_channel_prop']

for v in ['ip','time']:
    if v == 'time':
        v1 = ['app','channel']
    else:
        v1 = ['time','app','channel']
    
    for vv in v1:
        cnt = v+'_'+vv+'_cnt'
        attr =  v+'_'+vv+'_attr'
        prop = v+'_'+vv+'_prop'
        
        temp = ad.groupby([v,vv])['is_attributed'].count().reset_index(name='counts')
        temp.columns = [v,vv,cnt]
        ad = ad.merge(temp, on=[v,vv], how='left')
        
        temp = ad.groupby([v,vv])['is_attributed'].sum().reset_index(name='counts')
        temp.columns = [v,vv,attr]
        ad = ad.merge(temp, on=[v,vv], how='left')
        
        ad[prop]= np.nan
        ad[prop] = ad[attr] / ad[cnt]
        
        print(ad[[v,vv,cnt,attr,prop]].head(20))
        
ad['tot_vv_prop'] = np.nan
ad['tot_vv_prop'] = ad[var6].sum(axis=1)
print(ad['tot_vv_prop'].head(20))       
        

## Check correlation
feat = var1 + var2 + var3 + var4 + var5 + var6 + ['is_attributed']

print(ad[feat].corr(method='pearson'))
print(ad[feat].corr(method='spearman'))

pd.plotting.scatter_matrix(ad[var1 + ['is_attributed']], figsize=(15,15), alpha=.1, diagonal='kde')
pd.plotting.scatter_matrix(ad[var2 + ['is_attributed']], figsize=(15,15), alpha=.1, diagonal='kde')
pd.plotting.scatter_matrix(ad[var3 + ['is_attributed']], figsize=(15,15), alpha=.1, diagonal='kde')
pd.plotting.scatter_matrix(ad[var4 + ['is_attributed']], figsize=(15,15), alpha=.1, diagonal='kde')
pd.plotting.scatter_matrix(ad[var5 + ['is_attributed']], figsize=(15,15), alpha=.1, diagonal='kde')
pd.plotting.scatter_matrix(ad[var6 + ['is_attributed']], figsize=(15,15), alpha=.1, diagonal='kde')


## Reomve variables
for v in var1+var2+var4+var5:
    del ad[v]


## Save dataset
ad.to_csv('ad_modify_10m.csv', index=False)


## Check saved dataset
ad = pd.read_csv('ad_modify_10m.csv')
print(ad.columns)