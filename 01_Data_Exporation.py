
## 1. Data Exporaion
## Import library and data
import pandas as pd
import numpy as np

ad = pd.read_csv("train_sample.csv", parse_dates=['click_time'])


## Check dataset
print(ad.shape)
print(ad.columns)
print(ad.head(10))
print(ad.info())
print(ad.describe())


## Check the freqency of 'is_attributed' variable in ad
freq = ad['is_attributed'].value_counts(sort=False)
print(freq)


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
v = ['ip','app','device','os','channel','click_hour']
v1 = ['ip_cnt','app_cnt','device_cnt','os_cnt','channel_cnt','hour_cnt']
v2 = ['ip_attr','app_attr','device_attr','os_attr','channel_attr','hour_attr']
v3 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop','channel_attr_prop','hour_attr_prop']

for d,d1,d2,d3 in zip(v,v1,v2,v3):
    ad[d1] = np.nan
    ad[d1].fillna(ad.groupby(d)[d].transform('count'), inplace=True)
    print(ad[[d,d1]].head(20))

    ad[d2] = np.nan
    ad[d2].fillna(ad.groupby(d)['is_attributed'].transform('sum'), inplace=True)
    print(ad[[d,d2]].head(20))

    ad[d3] = np.nan
    ad[d3] = ad[d2] / ad[d1]
    print(ad[[d,d3]].head(20))
        
    ## Remove variable
    del ad[d1]
    del ad[d2]

ad['tot_prop'] = np.nan
ad['tot_prop'] = ad[v3].sum(axis=1)
print(ad['tot_prop'].head(20))


## Check correlation
feat = ['click_hour', 'ip_attr_prop', 'app_attr_prop', 'device_attr_prop', 'os_attr_prop', 
        'channel_attr_prop', 'hour_attr_prop', 'tot_prop', 'is_attributed']

pd.plotting.scatter_matrix(ad[feat], figsize=(15,15), alpha=.1)

print(ad[feat].corr(method='pearson'))
print(ad[feat].corr(method='spearman'))

    
## Save dataset
ad.to_csv('ad_dataset_modify.csv', index=False)


## Check saved dataset
ad = pd.read_csv('ad_dataset_modify.csv')
print(ad.columns)


## Set options
pd.set_option('display.max_columns', 200)