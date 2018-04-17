
## 1. Data Exporaion
## Import library and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ad = pd.read_csv("train_sample.csv")


## Check dataset
print(ad.shape)
print(ad.columns)
print(ad.head(10))
print(ad.info())
print(ad.describe())


from collections import Counter
Counter(ad['is_attributed'])


## Make derived values : ip_cnt, ip_attr, ip_attr_prob
ip_cnt = ad.groupby('ip')['ip'].count()
ip_attr = ad.groupby('ip')['is_attributed'].sum()
ip = pd.concat([ip_cnt, ip_attr], axis=1, keys=['click_count','attr_count'])
print(ip)

ad['ip_cnt'] = np.nan
ad['ip_cnt'].fillna(ad.groupby('ip')['ip'].transform('count'), inplace=True)
print(ad['ip_cnt'].head(20))

ad['ip_attr'] = np.nan
ad['ip_attr'].fillna(ad.groupby('ip')['is_attributed'].transform('sum'), inplace=True)
print(ad['ip_attr'].head(20))

ad['ip_attr_prob'] = np.nan
ad['ip_attr_prob'] = ad['ip_attr'] / ad['ip_cnt']
print(ad['ip_attr_prob'].head(20))


## Make derived values : app_cnt, app_attr, app_attr_prob 
app_cnt = ad.groupby('app')['app'].count()
app_attr = ad.groupby('app')['is_attributed'].sum()
app = pd.concat([app_cnt, app_attr], axis=1, keys=['click_count','attr_count'])
print(app)

ad['app_cnt'] = np.nan
ad['app_cnt'].fillna(ad.groupby('app')['app'].transform('count'), inplace=True)
print(ad['app_cnt'].head(20))

ad['app_attr'] = np.nan
ad['app_attr'].fillna(ad.groupby('app')['is_attributed'].transform('sum'), inplace=True)
print(ad['app_attr'].head(20))

ad['app_attr_prob'] = np.nan
ad['app_attr_prob'] = ad['app_attr'] / ad['app_cnt']
print(ad['app_attr_prob'])


## Make derived values : device_cnt, device_attr, device_attr_prob 
device_cnt = ad.groupby('device')['device'].count()
device_attr = ad.groupby('device')['is_attributed'].sum()
device = pd.concat([device_cnt, device_attr], axis=1, keys=['click_count','attr_count'])
print(device)

ad['device_cnt'] = np.nan
ad['device_cnt'].fillna(ad.groupby('device')['device'].transform('count'), inplace=True)
print(ad['device_cnt'].head(20))

ad['device_attr'] = np.nan
ad['device_attr'].fillna(ad.groupby('device')['is_attributed'].transform('sum'), inplace=True)
print(ad['device_attr'].head(20))

ad['device_attr_prob'] = np.nan
ad['device_attr_prob'] = ad['device_attr'] / ad['device_cnt']
print(ad['device_attr_prob'].head(20))


## Make derived values : os_cnt, os_attr, os_attr_prob
os_cnt = ad.groupby('os')['os'].count()
os_attr = ad.groupby('os')['is_attributed'].sum()
os = pd.concat([os_cnt, os_attr], axis=1, keys=['click_count','attr_count'])
print(os)

ad['os_cnt'] = np.nan
ad['os_cnt'].fillna(ad.groupby('os')['os'].transform('count'), inplace=True)
print(ad['os_cnt'].head(20))

ad['os_attr'] = np.nan
ad['os_attr'].fillna(ad.groupby('os')['is_attributed'].transform('sum'), inplace=True)
print(ad['os_attr'].head(20))

ad['os_attr_prob'] = np.nan
ad['os_attr_prob'] = ad['os_attr'] / ad['os_cnt']
print(ad['os_attr_prob'].head(20))


## Make derived values : channel_cnt, channel_attr, channel_attr_prob 
channel_cnt = ad.groupby('channel')['channel'].count()
channel_attr = ad.groupby('channel')['is_attributed'].sum()
channel = pd.concat([channel_cnt, channel_attr], axis=1, keys=['click_count','attr_count'])
print(channel)

ad['channel_cnt'] = np.nan
ad['channel_cnt'].fillna(ad.groupby('channel')['channel'].transform('count'), inplace=True)
print(ad['channel_cnt'].head(20))

ad['channel_attr'] = np.nan
ad['channel_attr'].fillna(ad.groupby('channel')['is_attributed'].transform('sum'), inplace=True)
print(ad['channel_attr'].head(20))

ad['channel_attr_prob'] = np.nan
ad['channel_attr_prob'] = ad['channel_attr'] / ad['channel_cnt']
print(ad['channel_attr_prob'].head(20))


## Make derived values : hour
print(ad[['click_time','attributed_time']].loc[ad['is_attributed'] == 1])

import datetime as dt

click_time = []
for x in ad['click_time']:
    time = dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    click_time.append(time)

ad['click_hour'] = np.nan
hour = []
for x in click_time:
    h = x.hour
    hour.append(h)
ad['click_hour'] = hour
print(ad[['click_time','click_hour']].head(10))


## Check correlation
print(ad[['ip_cnt', 'ip_attr', 'ip_attr_prob', 'app_cnt',
       'app_attr', 'app_attr_prob', 'device_cnt', 'device_attr',
       'device_attr_prob', 'os_cnt', 'os_attr', 'os_attr_prob', 'channel_cnt',
       'channel_attr', 'channel_attr_prob','is_attributed']].corr(method='pearson'))

## Save dataset
ad.to_csv('ad_dataset_modify.csv')

ad = pd.read_csv('ad_dataset_modify.csv')


pd.set_option('display.max_columns', 200)