
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
from collections import Counter
print(Counter(ad['is_attributed']))


## Make derived variable
## 'v'_cnt : frequency by 'v'
## 'v'_attr : the number of download by 'v'
## 'v'_attr_prop : the proporation of download by 'v'
v = ['ip','app','device','os','channel']
v1 = ['ip_cnt','app_cnt','device_cnt','os_cnt','channel_cnt']
v2 = ['ip_attr','app_attr','device_attr','os_attr','channel_attr']
v3 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop','channel_attr_prop']

for d,d1,d2,d3 in zip(v,v1,v2,v3):
    ad[d1] = np.nan
    ad[d1].fillna(ad.groupby(d)[d].transform('count'), inplace=True)
    print(ad[d1].head(20))

    ad[d2] = np.nan
    ad[d2].fillna(ad.groupby(d)['is_attributed'].transform('sum'), inplace=True)
    print(ad[d2].head(20))

    ad[d3] = np.nan
    ad[d3] = ad[d2] / ad[d1]
    print(ad[d3].head(20))


## Make derived values : click_hour
ad['click_hour'] = np.nan
hour = []
for x in ad.click_time:
    h = x.hour
    hour.append(h)
ad['click_hour'] = hour
print(ad[['click_time','click_hour']].head(10))


## Check correlation
print(ad[['ip_cnt', 'ip_attr', 'ip_attr_prop', 'app_cnt', 'app_attr', 'app_attr_prop', 
          'device_cnt', 'device_attr', 'device_attr_prop', 'os_cnt', 'os_attr', 'os_attr_prop', 
          'channel_cnt', 'channel_attr', 'channel_attr_prop', 'click_hour', 'is_attributed']].corr(method='pearson'))



## Remove variable
for i,j in zip(v1,v2):
    del ad[i]
    del ad[j]
del ad['click_time']
del ad['attributed_time']

    
## Save dataset
ad.to_csv('ad_dataset_modify.csv', index=False)


## Check saved dataset
ad = pd.read_csv('ad_dataset_modify.csv')
print(ad.columns)


## Set options
pd.set_option('display.max_columns', 200)