
## 2. Preprocessing
## Import library
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import gc


## Load data
data = pd.read_csv("data/merge.csv", parse_dates=['click_time'])
ip = pd.read_csv("blacklist/ip_black.csv")
app = pd.read_csv("blacklist/app_black.csv")
device = pd.read_csv("blacklist/device_black.csv")
os = pd.read_csv("blacklist/os_black.csv")
channel = pd.read_csv("blacklist/channel_black.csv")
hour = pd.read_csv("blacklist/hour_black.csv")
gc.collect()

print(data.head())
print(data.tail())


##    
def merge_black(feat):
    df = pd.read_csv('data/merge_' + feat + '.csv')
    df. head()
    
    black = pd.read_csv('blacklist/' + feat + '_black.csv')
    black.head()
    temp = black[[feat, 'gap','black_'+feat, 'rate']]
    df = df.merge(temp, on=feat, how='left')
    df.head()
    df.rename(columns={'gap':'gap_'+feat}, inplace = True)
    df.rename(columns={'rate':'rate_'+feat}, inplace = True)
    df.columns
    gc.collect()
    
    for name in ['gap_' + feat, 'black_' + feat, 'rate_' + feat]:
        temp = df[name].reset_index()
        temp.to_csv('data/merge_' + name + '.csv', index=False, columns=['index', name])
        temp.head()
    
merge_black('ip')
merge_black('app')
merge_black('device')
merge_black('os')
merge_black('channel')
merge_black('hour')


# Make a derived variable : click_gap
def click_gap(x):
    print(x, ' start...')
    temp = data.loc[data['ip'] == x, 'click_time'].reset_index()
    temp.sort_values(ascending=True, by='click_time', inplace=True)
    
    value = []
    value = list(temp['click_time'].iloc[:-1])
    value.insert(0, temp['click_time'].iloc[0])
    value = pd.to_datetime(value)
    
    time = []
    time = list(temp['click_time'])
    time = pd.to_datetime(time)
    
    temp['click_gap'] = np.nan
    temp['click_gap'] = value
    temp['click_time'] = time
    temp['click_gap'] = temp['click_time'] - temp['click_gap']
    temp['click_gap'] = temp['click_gap'].astype('timedelta64[s]')
    
    data.loc[list(temp['index']), 'click_gap'] = list(temp['click_gap'])
    gaps = list(temp['click_gap'])
    print(x, ' complete...')
    gc.collect()
    return gaps

index = data.ip.value_counts().reset_index()
data['click_gap'] = np.nan

print('count 1 : ', sum(index['ip'] == 1))  # 52,035
ones = index.loc[index['ip'] == 1]
index = index.loc[index['ip'] != 1]


gaps = index['index'].apply(lambda x: click_gap(x))
data.loc[list(ones['index']), 'click_gap'] = 0


data.head()
data.to_csv('merged_click_gap.csv', index=False)


## Divid data
del data['click_time']
gc.collect()

train = data.loc[data['click_id'].isnull()]
test = data.loc[data['click_id'].notnull()]

del data
del train['click_id']
gc.collect()

train.to_csv('train_modify.csv', index=False)
test.to_csv('test_modify.csv', index=False)

del test
gc.collect()


## Extract a sample
import random

for n in [10000000, 20000000,30000000,40000000,50000000]:
    idx = random.sample(range(len(train)),n)
    sample = train.iloc[idx]
    gc.collect()

    n = n / 1000000
    sample.to_csv('train_modify_' + str(n) + 'm.csv', index=False)
    
    del sample


##
def scatter_plot(feat, file_name):
    temp = train[feat]
    
    g = sns.pairplot(temp, 
                     hue='is_attributed', 
                     palette="husl",
                     plot_kws={'alpha':0.1})
    
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(60)
    
    g.fig.set_size_inches(10,8)
    plt.savefig('graph/'+file_name+'2.png')
    plt.show()
    gc.collect()

feat = ['is_attributed', 'gap_ip', 'black_ip']
scatter_plot(feat, 'scatter_plot_gap_black_ip')

feat = ['is_attributed', 'gap_app', 'black_app']
scatter_plot(feat, 'scatter_plot_gap_black_app')

feat = ['is_attributed', 'gap_device', 'black_device']
scatter_plot(feat, 'scatter_plot_gap_black_device')

feat = ['is_attributed', 'gap_os', 'black_os']
scatter_plot(feat, 'scatter_plot_gap_black_os')

feat = ['is_attributed', 'gap_channel', 'gap_hour']
scatter_plot(feat, 'scatter_plot_gap_channel_hour')


## check correlation
corr = round(train.corr(method='pearson'), 2)
plt.figure(figsize=(20,17))
sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
plt.savefig('graph/heatmap.png')
plt.show()
gc.collect()