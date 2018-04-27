
## 3. Test Data Preprocessing
## Import library and data
import pandas as pd
import numpy as np

ad = pd.read_csv('ad_modify_10m.csv')
print(ad.columns)

ad_test = pd.read_csv('test.csv', parse_dates=['click_time'])
print(ad_test.columns)


## Make derived variables of test data : hour,time
ad_test['hour'] = np.nan
ad_test['hour'] = ad_test['click_time'].dt.hour

ad_test['time'] = np.nan
ad_test['time'] = ad_test['hour'] // 4

print(ad_test[['click_time','hour','time']].head(20))


## Remove variables
del ad_test['click_id']
del ad_test['click_time']


## Make derived variables of test data
## 'v'_attr_prop : the proporation of download by 'v'
## 'v'_'vv'_prop : the proporation of download by 'v' and 'vv'
## tot_attr_prop : The total of 'v'_attr_prop
## tot_vv_prop : The total of 'v'_'vv'_prop
var = ['ip','app','device','os','channel','hour','time']
var1 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop',
        'channel_attr_prop','hour_attr_prop','time_attr_prop']
var2 = ['ip_time_prop','ip_app_prop','ip_channel_prop','time_app_prop','time_channel_prop']

for v,v1 in zip(var,var1):    
    temp = ad.groupby(v)[v1].mean().reset_index(name='counts')
    temp.columns = [v,v1]
    ad_test = ad_test.merge(temp, on=v, how='left')
    
    ## Fill missing values with mean
    ad_test[v1].fillna(ad[v1].median(), inplace=True)
    print('missing value of %s : %d' % (v1,ad_test[v1].isnull().sum()))
    
    print(ad_test[[v,v1]].head(20))

for v in ['ip','time']:
    if v == 'time':
        v1 = ['app','channel']
    else:
        v1 = ['time','app','channel']
    
    for vv in v1:      
        prop = v+'_'+vv+'_prop'
        
        temp = ad.groupby([v,vv])['is_attributed'].mean().reset_index(name='counts')
        temp.columns = [v,vv,prop]
        ad_test = ad_test.merge(temp, on=[v,vv], how='left')
        
        ad_test[prop].fillna(ad[prop].median(), inplace=True)
        print('missing value of %s : %d' % (prop,ad_test[prop].isnull().sum()))
        
        print(ad[[v,vv,prop]].head(20))

ad_test['tot_attr_prop'] = np.nan
ad_test['tot_attr_prop'] = ad_test[var1].sum(axis=1)
print(ad['tot_attr_prop'].head(20))

ad_test['tot_vv_prop'] = np.nan
ad_test['tot_vv_prop'] = ad_test[var2].sum(axis=1)
print(ad['tot_vv_prop'].head(20))


## Remove variables
for v in var:
    del ad[v]
    del ad_test[v]


## Save dataset
ad.to_csv('ad_modify2_10m.csv', index=False)
ad_test.to_csv('adtest_modify_10m.csv', index=False)


## Check saved dataset
ad = pd.read_csv('ad_modify2_10m.csv')
ad_test = pd.read_csv('adtest_modify_10m.csv')
print(ad.columns)
print(ad_test.columns)
print(ad.info())
print(ad_test.info())
