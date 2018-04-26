
## 4. Test Data Preprocessing
## Import library and data
import pandas as pd
import numpy as np

ad = pd.read_csv('ad_modify_10m.csv')
print(ad.columns)

ad_test = pd.read_csv('test.csv', parse_dates=['click_time'])
print(ad_test.columns)


## Make derived variable of test data : Click_hour
ad_test['click_hour'] = np.nan
ad_test['click_hour'] = ad_test['click_time'].dt.hour
print(ad_test[['click_time','click_hour']].head(10))


## Remove variable
del ad_test['click_id']
del ad_test['click_time']


## Make derived variable of test data
## 'v'_attr_prop : the proporation of download by 'v'
var1 = ['ip','app','device','os','channel','click_hour']
var2 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop','channel_attr_prop','hour_attr_prop']

for v1,v2 in zip(var1,var2):
    temp = ad.groupby(v1)[v2].mean().reset_index(name='counts')
    temp.columns = [v1,v2]
    ad_test = ad_test.merge(temp, on=v1, how='left')
    print(ad_test[[v1,v2]].head(20))


## Fill missing values with mean
for v in var2:
    ad_test[v].fillna(ad[v].mean(), inplace=True)
    print(ad_test[v].isnull().sum())


## Remove variable
for v in var1:
    del ad[v]
    del ad_test[v]
    
for v in ['ip_cnt','app_cnt','device_cnt','os_cnt','channel_cnt','hour_cnt',
          'ip_attr','app_attr','device_attr','os_attr','channel_attr','hour_attr']:
    del ad[v]


## Make derived variable of test data
## tot_prop : ip_attr_prop + app_attr_prop + device_prop + os_attr_prop + channel_attr_prop + hour_attr_prop
ad_test['tot_prop'] = np.nan
ad_test['tot_prop'] = ad_test[var2].sum(axis=1)
print(ad['tot_prop'].head(20))


## Save dataset
ad.to_csv('ad_modify2_10m.csv', index=False)
ad_test.to_csv('adtest_modify_10m.csv', index=False)


## Check saved dataset
ad_test = pd.read_csv('adtest_modify_10m.csv')
print(ad_test.columns)
print(ad_test.info())
