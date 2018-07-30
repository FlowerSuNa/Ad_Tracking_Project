
## 1. EDA
## Import library
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import gc


## Load data
train = pd.read_csv("data/train.csv", parse_dates=['click_time', 'attributed_time'])
test = pd.read_csv('data/test.csv', parse_dates=['click_time'])
gc.collect()


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
# --- 0 : 18,447,044
# --- 1 : 456,846


## Check 'click_time'
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


## Draw time series of train data click time
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


## Draw time series of test data click time
temp = test['click_time']
temp.index = test['click_time']
temp = temp.resample('10T').count()

plt.figure(figsize=(10,5))
plt.title('click time (10 minute bins) of test data')
plt.plot(temp.index, temp, 'g')
plt.xticks(rotation=60, fontsize="small")
plt.savefig('graph/test_click_time.png')
plt.show()
gc.collect()


## Draw time series of click time and attributed time
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
train['hour'] = np.nan
train['hour'] = train['click_time'].dt.hour

test['hour'] = np.nan
test['hour'] = test['click_time'].dt.hour
gc.collect()

print(train[['click_time', 'hour']].head(10))
print(train[['click_time', 'hour']].tail(10))

print(test[['click_time', 'hour']].head(10))
print(test[['click_time', 'hour']].tail(10))


## Draw hour bar graphs
plt.figure(figsize=(15,10))

plt.subplot(2,1,1)
plt.title('click count per hour in train data')
sns.countplot('hour', data=train, linewidth=0)

plt.subplot(2,1,2)
plt.title('click count per hour in test data')
sns.countplot('hour', data=test, linewidth=0)

plt.savefig('graph/hour_click_count.png')
plt.show()
gc.collect()


## Draw hour and download bar graphs
plt.figure(figsize=(15,15))

plt.subplot(3,1,1)
plt.title('click count per hour in train data')
sns.countplot('hour', data=train)

plt.subplot(3,1,2)
plt.title('download count per hour')
sns.barplot('hour', 'is_attributed', data=train, estimator=sum, ci=None)

plt.subplot(3,1,3)
plt.title('download rate by per hour')
sns.barplot('hour', 'is_attributed', data=train, ci=None)

plt.savefig('graph/hour_download_rate.png')
plt.show()
gc.collect()


## Merge train data and test data
del train['attributed_time']
test['is_attributed'] = 0
data = pd.concat([train, test])

print('merged data shape : ', data.shape)       # (203694359, 9)
print(data.head(10))

del train
del test
gc.collect()
data.to_csv('data/merge.csv', index=False)


## Make black list
def make_black_list(v):
    temp = data[v].value_counts().reset_index()
    temp.columns = [v,'count']
    temp.sort_values(ascending=False, by='count', inplace=True)
    
    temp2 = data.groupby(v)['is_attributed'].sum().reset_index()
    temp2.columns = [v,'download']
    temp = temp.merge(temp2, on=v, how='left')
    
    print('sort by count')
    print(temp.head(30))
    print(temp.tail(30))
    print()
    
    temp.sort_values(ascending=False, by='download', inplace=True)
    
    print('sort by download')
    print(temp.head(30))
    print(temp.tail(30))
    print()
    
    temp['gap'] = temp['count'] - temp['download']
    temp.sort_values(ascending=False, by='gap', inplace=True)
    
    print('sort by gap')
    print(temp.head(30))
    print(temp.tail(30))
    print()
    
    temp['rate'] = temp['download'] / temp['count']
    temp.sort_values(ascending=False, by='rate', inplace=True)
    
    print('sort by rate')
    print(temp.head(30))
    print(temp.tail(30))
    print()
    
    black_boundary = temp['count'].median() + 10
    print('black list boundary : ', black_boundary)
    
    temp['black_' + v] = 0
    temp.loc[(temp['count'] > black_boundary) & (temp['rate'] == 0), 'black_' + v] = 1
    temp.sort_values(by=v, inplace=True)
    
    print('check black list')
    print(temp.head(30))
    print(temp.tail(30))
    print('count : ', temp['black_' + v].sum())
    
    temp.to_csv('blacklist/' + v + '_download.csv', index=False)
    return temp

ip = make_black_list('ip')              # count : 34,770
app = make_black_list('app')            # count : 178
device = make_black_list('device')      # count : 62
os = make_black_list('os')              # count : 170
chennel = make_black_list('channel')    # count : 0
hour = make_black_list('hour')          # count : 0


## Draw bar graphs
def barplot(data, v):
    temp1 = data.head(30)
    temp2 = data.tail(30)
    plt.figure(figsize=(20,15))
    
    plt.subplot(3,2,1)
    plt.title('click count per ' + v + ' (top)')
    sns.barplot(v, 'count', data=temp1, linewidth=0)
    plt.xticks(rotation=90, fontsize="small")
    
    plt.subplot(3,2,2)
    plt.title('(bottom)')
    sns.barplot(v, 'count', data=temp2, linewidth=0)
    plt.xticks(rotation=90, fontsize="small")
    
    plt.subplot(3,2,3)
    plt.title('download count per ' + v + ' (top)')
    sns.barplot(v, 'download', data=temp1, linewidth=0, estimator=sum, ci=None)
    plt.xticks(rotation=90, fontsize="small")
    
    plt.subplot(3,2,4)
    plt.title('(bottom)')
    sns.barplot(v, 'download', data=temp2, linewidth=0, estimator=sum, ci=None)
    plt.xticks(rotation=90, fontsize="small")
        
    plt.subplot(3,2,5)
    plt.title('download rate per ' + v + ' (top)')
    sns.barplot(v, 'rate', data=temp1, linewidth=0, ci=None)
    plt.xticks(rotation=90, fontsize="small")
    
    plt.subplot(3,2,6)
    plt.title('(bottom)')
    sns.barplot(v, 'rate', data=temp2, linewidth=0, ci=None)
    plt.xticks(rotation=90, fontsize="small")
    
    plt.savefig('graph/' + v + '_download_rate.png')
    plt.show()
    gc.collect()
    
ip.sort_values(ascending=False, by='count', inplace=True)
barplot(ip, 'ip')

app.sort_values(ascending=False, by='count', inplace=True)
barplot(app, 'app')

device.sort_values(ascending=False, by='count', inplace=True)
barplot(device, 'device')

os.sort_values(ascending=False, by='count', inplace=True)
barplot(os, 'os')

chennel.sort_values(ascending=False, by='count', inplace=True)
barplot(chennel, 'chennel')


## Draw scatter plots
def scatter_plot(feat, file_name):
    temp = data[feat].loc[data['click_id'].isnull()]
    
    g = sns.pairplot(temp, 
                     hue='is_attributed', 
                     palette="husl",
                     plot_kws={'alpha':0.1})
    
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(90)
            
    g.fig.set_size_inches(10,10)
    plt.savefig('graph/'+file_name+'.png')
    plt.show()
    gc.collect()
    
feat = ['is_attributed', 'ip']
scatter_plot(feat, 'scatter_plot_ip')

feat = ['is_attributed', 'app']
scatter_plot(feat, 'scatter_plot_app')

feat = ['is_attributed', 'device']
scatter_plot(feat, 'scatter_plot_device')

feat = ['is_attributed', 'os']
scatter_plot(feat, 'scatter_plot_os')

feat = ['is_attributed', 'chennel']
scatter_plot(feat, 'scatter_plot_chennel')

feat = ['is_attributed', 'hour']
scatter_plot(feat, 'scatter_plot_hour')