
## 2-2. Test Data Preprocessing
## Import library and data
import pandas as pd
import numpy as np
import gc

ad = pd.read_csv('train_modify1.csv')
print(ad.columns)

ad_test = pd.read_csv('test.csv', parse_dates=['click_time'])
print(ad_test.columns)


## Make a derived variable 'hour'
ad_test['hour'] = np.nan
ad_test['hour'] = ad_test['click_time'].dt.hour
gc.collect()

print(ad_test[['click_time','hour']].head(10))
print(ad_test[['click_time','hour']].tail(10))


## Remove variables
del ad_test['click_id']
del ad_test['click_time']
gc.collect()


## Make derived variables of test data
## 'v'_attr_prop : download proportion by 'v'
## tot_attr_prop : the total of 'v'_attr_prop
var = ['ip','app','device','os','channel','hour']
var1 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop','channel_attr_prop','hour_attr_prop']

for v,v1 in zip(var,var1):    
    temp = ad.groupby(v)[v1].mean().reset_index(name='counts')
    temp.columns = [v,v1]
    ad_test = ad_test.merge(temp, on=v, how='left')
    gc.collect()
    
    ## Fill missing values with mean
    if ad_test[v1].isnull().sum() != 0:
        print('missing value of %s : %d' % (v1,ad_test[v1].isnull().sum()))
        ad_test[v1].fillna(ad[v1].mean(), inplace=True)
        print('missing value of %s : %d' % (v1,ad_test[v1].isnull().sum()))
    
    print(ad_test[[v,v1]].head(10))
    print(ad_test[[v,v1]].tail(10))
    
ad_test['tot_attr_prop'] = np.nan
ad_test['tot_attr_prop'] = ad_test[var1].sum(axis=1)
gc.collect()

print(ad_test['tot_attr_prop'].head(10))
print(ad_test['tot_attr_prop'].tail(10))


## 'v'_'vv'_prop : download proportion by 'v' and 'vv'
## tot_vv_prop : the total of 'v'_'vv'_prop
var2 = ['ip_hour_prop','ip_app_prop','ip_channel_prop','hour_app_prop','hour_channel_prop']

for v in ['ip','hour']:
    if v == 'hour':
        v1 = ['app','channel']
    else:
        v1 = ['hour','app','channel']
    
    for vv in v1:      
        prop = v+'_'+vv+'_prop'
        
        temp = ad.groupby([v,vv])[prop].mean().reset_index(name='counts')
        temp.columns = [v,vv,prop]
        ad_test = ad_test.merge(temp, on=[v,vv], how='left')
        gc.collect()
        
        ## Fill missing values with mean
        if ad_test[prop].isnull().sum() != 0:
            print('missing value of %s : %d' % (prop,ad_test[prop].isnull().sum()))
            ad_test[prop].fillna(ad[prop].mean(), inplace=True)
            print('missing value of %s : %d' % (prop,ad_test[prop].isnull().sum()))
        
        print(ad_test[[v,vv,prop]].head(10))
        print(ad_test[[v,vv,prop]].tail(10))

ad_test['tot_vv_prop'] = np.nan
ad_test['tot_vv_prop'] = ad_test[var2].sum(axis=1)
gc.collect()

print(ad_test['tot_vv_prop'].head(10))
print(ad_test['tot_vv_prop'].tail(10))


## Save dataset
ad_test.to_csv('test_modify1.csv', index=False)
