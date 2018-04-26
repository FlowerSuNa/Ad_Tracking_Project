
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


## Check correlation
feat = ['ip','app','device','os','channel']

pd.plotting.scatter_matrix(ad[feat], figsize=(15,15), alpha=.1)

print(ad[feat].corr(method='pearson'))
print(ad[feat].corr(method='spearman'))


## Set options
pd.set_option('display.max_columns', 200)