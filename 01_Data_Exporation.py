
## 1. Data Exporaion
## Import library and data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ad = pd.read_csv("train.csv", parse_dates=['click_time'])


## Create functions
def barplot(x, y, data):
    plt.figure(figsize=(15,15))

    plt.subplot(3,1,1)
    plt.title('click count by ' + x)
    sns.countplot(x, data=data)
    
    plt.subplot(3,1,2)
    plt.title('download count by ' + x)
    sns.barplot(x, y, data=data, estimator=sum, ci=None)
    
    plt.subplot(3,1,3)
    plt.title('download rate by ' + x)
    sns.barplot(x, y, data=data, ci=None)
    
    plt.savefig('graph/barplot_' + x + '.png')
    plt.show()
    plt.close()
    
def barplot_r(x, y, data):
    plt.figure(figsize=(25,20))

    plt.subplot(3,1,1)
    plt.title('click count by ' + x)
    sns.countplot(x, data=data)
    plt.xticks(rotation=90, fontsize="small")
    
    plt.subplot(3,1,2)
    plt.title('download count by ' + x)
    sns.barplot(x, y, data=data, estimator=sum, ci=None)
    plt.xticks(rotation=90, fontsize="small")
    
    plt.subplot(3,1,3)
    plt.title('download rate by ' + x)
    sns.barplot(x, y, data=data, ci=None)
    plt.xticks(rotation=90, fontsize="small")

    plt.savefig('graph/barplot_' + x + '.png')  
    plt.show()
    plt.close()


## Check dataset
print(ad.shape)
print(ad.columns)
print(ad.head(10))
print(ad.info())
print(ad.describe())


## Check download frequency
freq = ad['is_attributed'].value_counts(sort=False)
print(freq)


## Check the number of downloads over time
temp = ad['is_attributed']
temp.index = ad['click_time']
temp = temp.resample('H').sum()

plt.figure(figsize=(10,5))
plt.title('The number of download by click time')
plt.plot(temp.index, temp)
plt.savefig('graph/timeplot.png')
plt.show()


## Check click count, download count, download rate per hour
ad['hour'] = np.nan
ad['hour'] = ad['click_time'].dt.hour
barplot('hour', 'is_attributed', ad)


## Draw barplots
barplot_r('app', 'is_attributed', ad)
barplot_r('device', 'is_attributed', ad)
barplot_r('os', 'is_attributed', ad)
barplot_r('channel', 'is_attributed', ad)


## Check correlation
feat = ['ip','app','device','os','channel','hour','is_attributed']
print(ad[feat].corr(method='pearson'))
pd.plotting.scatter_matrix(ad[feat], figsize=(15,15), alpha=.1)
plt.savefig('graph/scatterplot.png')
plt.show()



## Set options
# pd.set_option('display.max_columns', 200)

## countplot
## https://seaborn.pydata.org/generated/seaborn.countplot.html

## barplot
## https://seaborn.pydata.org/generated/seaborn.barplot.html