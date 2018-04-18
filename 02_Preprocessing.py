
## 2. Data Analysis
## Import library and data
import pandas as pd
import numpy as np

ad = pd.read_csv('ad_dataset_modify.csv')
print(ad.columns)

ad_test = ad = pd.read_csv('test.csv', parse_dates=['click_time'])
print(ad_test.columns)


##
feat1 = ['ip','app','device','os','channel']
feat2 = ['ip_attr_prop','ip_attr_prop','ip_attr_prop','ip_attr_prop','ip_attr_prop']

for f1, f2 in zip(feat1,feat2):
    ad_test[f2] = np.nan
    a = ad.groupby(f1)[f2].mean()

    for x,y in zip(a.index,a):
        ad_test[f2].loc[ad_test[f1] == x] = y
    print(ad_test[[f1,f2]].head(20))


##
ad_test.fillna(0, inplace=True)


##
ad_test['click_hour'] = np.nan
hour = []
for x in ad_test.click_time:
    h = x.hour
    hour.append(h)
ad_test['click_hour'] = hour
print(ad_test[['click_time','click_hour']].head(10))


## Remove variable
del ad_test['click_time']


## Save dataset
ad_test.to_csv('adtest_dataset_modify.csv', index=False)


## Check saved dataset
ad_test = pd.read_csv('adtest_dataset_modify.csv', parse_dates=['click_time'])
print(ad_test.columns)
print(ad_test.info())
