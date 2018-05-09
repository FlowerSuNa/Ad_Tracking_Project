
## 1. Data Exporaion
## Import library and data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ad = pd.read_csv("train_sample.csv", parse_dates=['click_time'])


## Create functions
def barplot(x, y, data):
    plt.figure(figsize=(15,15))

    plt.subplot(3,1,1)
    plt.title('the frequency of ' + x)
    sns.countplot(x, data=data)
    
    plt.subplot(3,1,2)
    plt.title('the number of download by ' + x)
    sns.barplot(x, y, data=data, estimator=sum, ci=None)
    
    plt.subplot(3,1,3)
    plt.title('the proporation of download by ' + x)
    sns.barplot(x, y, data=data, ci=None)
    
    plt.show()


## Check dataset
print(ad.shape)
print(ad.columns)
print(ad.head(10))
print(ad.info())
print(ad.describe())


## Check download frequency
freq = ad['is_attributed'].value_counts(sort=False)
print(freq)


## Check the number of download over time
temp = ad['is_attributed']
temp.index = ad['click_time']
temp = temp.resample('H').sum()

plt.figure(figsize=(10,10))
plt.title('The number of download by click time')
plt.plot(temp.index, temp)
plt.show()


## Check download per hour
ad['hour'] = np.nan
ad['hour'] = ad['click_time'].dt.hour
barplot('hour', 'is_attributed', ad)


## Draw barplots
barplot('ip', 'is_attributed', ad)
barplot('app', 'is_attributed', ad)
barplot('device', 'is_attributed', ad)
barplot('os', 'is_attributed', ad)
barplot('channel', 'is_attributed', ad)


freq = ad.ip.value_counts(sort=True)
freq[:30]


## Check correlation
feat = ['ip','app','device','os','channel']

pd.plotting.scatter_matrix(ad[feat], figsize=(15,15), alpha=.1)

print(ad[feat].corr(method='pearson'))
print(ad[feat].corr(method='spearman'))


## Set options
pd.set_option('display.max_columns', 200)



## countplot
## https://seaborn.pydata.org/generated/seaborn.countplot.html

## varplot
## https://seaborn.pydata.org/generated/seaborn.barplot.html