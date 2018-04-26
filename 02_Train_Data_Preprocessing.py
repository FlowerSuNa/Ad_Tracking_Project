
## 2. Train Data Preprocessing
## Import library and data
import pandas as pd
import numpy as np

ad = pd.read_csv("train_sample.csv", parse_dates=['click_time'])
print(ad.shape)
print(ad.columns)


## Make derived values : click_hour
ad['click_hour'] = np.nan
ad['click_hour'] = ad['click_time'].dt.hour
print(ad[['click_time','click_hour']].head(10))


## Remove variable
del ad['click_time']
del ad['attributed_time']


## Make derived variable
## 'v'_cnt : frequency by 'v'
## 'v'_attr : the number of download by 'v'
## 'v'_attr_prop : the proporation of download by 'v'
## tot_prop : ip_attr_prop + app_attr_prop + device_prop + os_attr_prop + channel_attr_prop + hour_attr_prop
var = ['ip','app','device','os','channel','click_hour']
var1 = ['ip_cnt','app_cnt','device_cnt','os_cnt','channel_cnt','hour_cnt']
var2 = ['ip_attr','app_attr','device_attr','os_attr','channel_attr','hour_attr']
var3 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop','channel_attr_prop','hour_attr_prop']

for v,v1,v2,v3 in zip(var,var1,var2,var3):
    temp = ad[v].value_counts().reset_index(name='counts')
    temp.columns = [v,v1]
    ad = ad.merge(temp, on=v, how='left')
    print(ad[[v,v1]].head(20))

    temp = ad.groupby(v)['is_attributed'].sum().reset_index(name='counts')
    temp.columns = [v,v2]
    ad = ad.merge(temp, on=v, how='left')
    print(ad[[v,v2]].head(20))

    ad[v3] = np.nan
    ad[v3] = ad[v2] / ad[v1]
    print(ad[[v,v3]].head(20))

ad['tot_prop'] = np.nan
ad['tot_prop'] = ad[var3].sum(axis=1)
print(ad['tot_prop'].head(20))


## Check correlation
feat = var1 + var2 + var3 + ['is_attributed']

print(ad[feat].corr(method='pearson'))
print(ad[feat].corr(method='spearman'))

pd.plotting.scatter_matrix(ad[v1 + ['is_attributed']], figsize=(15,15), alpha=.1)
pd.plotting.scatter_matrix(ad[v2 + ['is_attributed']], figsize=(15,15), alpha=.1)
pd.plotting.scatter_matrix(ad[v3 + ['is_attributed']], figsize=(15,15), alpha=.1)

    
## Save dataset
ad.to_csv('ad_modify_10m.csv', index=False)


## Check saved dataset
ad = pd.read_csv('ad_modify_10m.csv')
print(ad.columns)