
## 1. EDA
## Import library
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import gc


## Load data
train = pd.read_csv("train_sample.csv", parse_dates=['click_time', 'attributed_time'])
test = pd.read_csv('test.csv', parse_dates=['click_time'])
gc.collect()


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
print('train data shape : ', train.shape)   # (184903890, 8)
print('test data shape : ', test.shape)     # (18790469, 7)

print('train data columns : \n', train.columns)
print('test data columns : \n', test.columns)

print('train data info. : \n', train.info())
print('test data info. : \n', test.info())

print('train data describe : \n', train.describe())
print('test data describe : \n', test.describe())

print('train data head : \n', train.head(10))
print('test data head : \n', test.head(10))


## Check download frequency
freq = train['is_attributed'].value_counts(sort=False)
print('train data download frequency : \n', freq)
# --- 0 : 18447044
# --- 1 : 456846


##
print('year count of train data : \n', train['click_time'].dt.year.value_counts())
# --- 2017 : 184,903,890

print('momth count of train data : \n', train['click_time'].dt.month.value_counts())
# --- 11 : 184,903,890

print('day count of train data : \n', train['click_time'].dt.day.value_counts())
# --- 6 : 9,308,568
# --- 7 : 59,633,310
# --- 8 : 62,945,075
# --- 9 : 53,016,937

print('year count of test data : \n', test['click_time'].dt.year.value_counts())
# --- 2017 : 18,790,469

print('momth count of test data : \n', test['click_time'].dt.month.value_counts())
# --- 11 : 18,790,469

print('day count of test data : \n', test['click_time'].dt.day.value_counts())
# --- 10  :18,790,469


## Check train data click time
temp = train['click_time']
temp.index = train['click_time']
temp = temp.resample('10T').count()

plt.figure(figsize=(10,5))
plt.title('click time (10 minute bins) of train data')
plt.plot(temp.index, temp, 'g')
plt.xticks(label=[])
plt.savefig('graph/train_click_time.png')
plt.show()
gc.collect()


## Check test data click time
temp = test['click_time']
temp.index = test['click_time']
temp = temp.resample('10T').count()

plt.figure(figsize=(10,5))
plt.title('click time (10 minute bins) of test data')
plt.plot(temp.index, temp, 'g')
plt.savefig('graph/test_click_time.png')
plt.show()
gc.collect()


## Check click time and attributed time
temp1 = train['is_attributed']
temp1.index = train['click_time']
temp1 = temp1.resample('10T').sum()

temp2 = train['is_attributed']
temp2.index = train['attributed_time']
temp2 = temp2.resample('10T').sum()

plt.figure(figsize=(10,5))
plt.title('click time and attributed time')
plt.plot(temp1.index, temp1, 'g', label='click time')
plt.plot(temp2.index, temp2, 'r', label='attributed time')
plt.legend(loc='lower right', fontsize='small')
plt.savefig('graph/train_click_download.png')
plt.show()
gc.collect()


## Make a derived variable : hour
train['hour'] = train['click_time'].dt.hour
test['hour'] = test['click_time'].dt.hour
gc.collect()

print(train[['click_time', 'hour']].head(10))
print(train[['click_time', 'hour']].tail(10))

print(test[['click_time', 'hour']].head(10))
print(test[['click_time', 'hour']].tail(10))


##
plt.figure(figsize=(10,15))

plt.subplot(2,1,1)
plt.title('click count per hour in train data')
sns.countplot('hour', data=train)

plt.subplot(2,1,1)
plt.title('click count per hour in test data')
sns.countplot('hour', data=test)

plt.savefig('graph/hour_cilck_count.png')
plt.show()
gc.collect()


##
plt.figure(figsize=(15,15))

plt.subplot(3,1,1)
plt.title('click count per hour in train data')
sns.countplot('hour', train)

plt.subplot(3,1,2)
plt.title('download count per hour')
sns.barplot('hour', 'is_attributed', data=train, estimator=sum, ci=None)

plt.subplot(3,1,3)
plt.title('download rate by per hour')
sns.barplot('hour', 'is_attributed', data=train, ci=None)

plt.savefig('graph/hour_download_rate.png')
plt.show()
gc.collect()


##
temp = train['ip'].value_counts()
temp.sort()




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

