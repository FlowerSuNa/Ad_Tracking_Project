
# 2. Train Data Preprocessing

### Import library and data


```python
import pandas as pd
import numpy as np

ad = pd.read_csv("train_sample.csv", parse_dates=['click_time'])
```

### Check data


```python
print(ad.shape)
print(ad.columns)
```

### Make a derived variable : click_hour


```python
ad['click_hour'] = np.nan
ad['click_hour'] = ad['click_time'].dt.hour
print(ad[['click_time','click_hour']].head(10))
```

### Remove variables : click_time, attributed_time


```python
del ad['click_time']
del ad['attributed_time']
```

### Make derived variables

* 'v'_cnt : frequency by 'v'
* 'v'_attr : the number of download by 'v'
* 'v'_attr_prop : the proporation of download by 'v'
* tot_prop : ip_attr_prop + app_attr_prop + device_prop + os_attr_prop + channel_attr_prop + hour_attr_prop


```python
var = ['ip','app','device','os','channel','click_hour']
var1 = ['ip_cnt','app_cnt','device_cnt','os_cnt','channel_cnt','hour_cnt']
var2 = ['ip_attr','app_attr','device_attr','os_attr','channel_attr','hour_attr']
var3 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop','channel_attr_prop','hour_attr_prop']

for v,v1,v2,v3 in zip(var,var1,var2,var3):
    temp = ad[v].value_counts().reset_index(name='counts')
    temp.columns = [v,v1]
    ad = ad.merge(temp, on=v, how='left')

    temp = ad.groupby(v)['is_attributed'].sum().reset_index(name='counts')
    temp.columns = [v,v2]
    ad = ad.merge(temp, on=v, how='left')

    ad[v3] = np.nan
    ad[v3] = ad[v2] / ad[v1]
    
    print(ad[[v,v1,v2,v3]].head(20))
```


```python
ad['tot_prop'] = np.nan
ad['tot_prop'] = ad[var3].sum(axis=1)
print(ad['tot_prop'].head(20))
```

### Check correlation


```python
feat = var1 + var2 + var3 + ['is_attributed']
```

pearson


```python
print(ad[feat].corr(method='pearson'))
```

spearman


```python
print(ad[feat].corr(method='spearman'))
```

scatter plot


```python
pd.plotting.scatter_matrix(ad[var1 + ['is_attributed']], figsize=(15,15), alpha=.1)
```


```python
pd.plotting.scatter_matrix(ad[var2 + ['is_attributed']], figsize=(15,15), alpha=.1)
```


```python
pd.plotting.scatter_matrix(ad[var3 + ['is_attributed']], figsize=(15,15), alpha=.1)
```
