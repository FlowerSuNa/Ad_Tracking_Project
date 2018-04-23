
## 2. Preprocessing
## Import library and data
import pandas as pd
import numpy as np

ad = pd.read_csv('ad_dataset_modify.csv')
print(ad.columns)

ad_test = pd.read_csv('test.csv', parse_dates=['click_time'])
print(ad_test.columns)


## Make derived variable of test data : Click_hour
ad_test['click_hour'] = np.nan
ad_test['click_hour'] = ad_test['click_time'].dt.hour
print(ad_test[['click_time','click_hour']].head(10))


## Make derived variable of test data
## 'v'_attr_prop : the proporation of download by 'v'
feat1 = ['ip','app','device','os','channel','click_hour']
feat2 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop','channel_attr_prop','hour_attr_prop']

for f1, f2 in zip(feat1,feat2):
    ad_test[f2] = np.nan
    a = ad.groupby(f1)[f2].mean()

    for x,y in zip(a.index,a):
        ad_test.loc[ad_test[f1] == x, f2] = y
    print(ad_test[[f1,f2]].head(20))


## Fill missing values of ip_attr_prop with mean of 
for f in feat2:
    ad_test[f].fillna(ad[f].mean(), inplace=True)
    print(ad_test[f].isnull().sum())


## Make derived variable of test data
## tot_prop : ip_attr_prop + app_attr_prop + device_prop + os_attr_prop + channel_attr_prop + hour_attr_prop
ad_test['tot_prop'] = np.nan
ad_test['tot_prop'] = ad_test[feat2].sum(axis=1)
print(ad['tot_prop'].head(20))


## Remove variable
del ad_test['click_id']
del ad_test['click_time']


## Save dataset
ad_test.to_csv('adtest_dataset_modify.csv', index=False)


## Check saved dataset
ad_test = pd.read_csv('adtest_dataset_modify.csv')
print(ad_test.columns)
print(ad_test.info())
