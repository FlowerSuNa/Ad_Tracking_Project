
## 1. Data Exporaion
## Import library and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ad = pd.read_csv("train_sample.csv")


## Check dataset
print(ad.shape)
print(ad.head(10))
print(ad.info())
print(ad.describe())


## Make derived values : ip_attr_prob, app_attr_prob, device_attr_prob, os_attr_prob, channel_attr_prob 
## from collections import counter

ip_cnt = ad.groupby('ip')['ip'].count()
ip_attr = ad.groupby('ip')['is_attributed'].sum()
ip = pd.concat([ip_cnt, ip_attr], axis=1, keys=['click_count','attr_count'])
print(ip)

ad['ip_cnt'] = np.nan
ad['ip_cnt'].fillna(ad.groupby('ip')['ip'].transform('count'), inplace=True)

ad['ip_attr'] = np.nan
ad['ip_attr'].fillna(ad.groupby('ip')['is_attributed'].transform('sum'), inplace=True)


ad['ip_attr_prob'] = np.nan
ad['ip_attr_prob'] = ad['ip_attr'] / ad['ip_cnt']

app_cnt = ad.groupby('app')['app'].count()
app_attr = ad.groupby('app')['is_attributed'].sum()
app = pd.concat([app_cnt, app_attr], axis=1, keys=['click_count','attr_count'])
print(app)

device_cnt = ad.groupby('device')['device'].count()
device_attr = ad.groupby('device')['is_attributed'].sum()
device = pd.concat([device_cnt, device_attr], axis=1, keys=['click_count','attr_count'])
print(device)

os_cnt = ad.groupby('os')['os'].count()
os_attr = ad.groupby('os')['is_attributed'].sum()
os = pd.concat([os_cnt, os_attr], axis=1, keys=['click_count','attr_count'])
print(os)

channel_cnt = ad.groupby('channel')['channel'].count()
channel_attr = ad.groupby('channel')['is_attributed'].sum()
channel = pd.concat([channel_cnt, channel_attr], axis=1, keys=['click_count','attr_count'])
print(channel)


## Make derived values : day, hour
print(ad[['click_time','attributed_time']].loc[ad['is_attributed'] == 1])

import datetime as dt

click_time = []
for x in ad['click_time']:
    time = dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    click_time.append(time)

ad['click_day'] = np.nan
day = []
for x in click_time:
    d = x.day
    day.append(d)
ad['click_day'] = day
print(ad[['click_time','click_day']].head(10))

ad['click_hour'] = np.nan
hour = []
for x in click_time:
    h = x.hour
    hour.append(h)
ad['click_hour'] = hour
print(ad[['click_time','click_hour']].head(10))


## Save dataset
ad.to_csv('ad_dataset_modify.csv')

ad = pd.read_csv('ad_dataset')



ad.columns