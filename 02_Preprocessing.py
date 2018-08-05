
## 2. Preprocessing
## Import library
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import gc

train_test_boundary = 184903890


## Make derived variable : gap, black and rate per (ip, app, device, os, channel, hour)
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

for feat in ['ip', 'app', 'device', 'os', 'channel', 'hour']:
    merge_black(feat)    


## Make a derived variable : click_gap
data = pd.read_csv('data/merge_ip.csv', index_col=['index'])
temp = pd.read_csv('data/merge_click_time.csv', index_col=['index'])

data = pd.concat([data, temp], axis=1)
data = data.reset_index()
data.sort_values(by=['ip', 'click_time'], inplace=True)

data['i'] = np.arange(0, len(data))
data = data.set_index('i')

data['click_time'] = pd.to_datetime(data['click_time'])

temp = data.loc[:len(data) - 1, 'click_time']
temp = temp.reset_index()

temp['i'] = np.arange(1, len(temp)+1)
temp =  temp.set_index('i')
temp.loc[0] = np.nan

data['click_gap'] = np.nan
data['click_gap'] = temp
data['click_gap'] = data['click_time'] - data['click_gap']

data['i'] = np.arange(0, len(data))
index = data.groupby('ip')['i'].min()
index = list(index)

del data['i']

data.loc[index, 'click_gap'] = np.nan

data['click_gap'] = data['click_gap'].astype('timedelta64[s]')
data.sort_values(by='index', ascending=True, inplace=True)
data = data.set_index('index')

temp = data['click_gap'].reset_index()
temp.to_csv('data/merge_click_gap.csv', index=False)
  




## Concat data
data = pd.read_csv('data/merge_is_attributed.csv', index_col=['index'])

for name in ['ip','app','device','os','channel','hour']:
    feat = [name, 'gap_' + name, 'black_' + name, 'rate_' + name]
    for f in feat:
        temp = pd.read_csv('data/merge_' + f + '.csv', index_col=['index'])
        data = pd.concat([data, temp], axis=1)
        del temp
        gc.collect()

temp = pd.read_csv('data/merge_click_gap.csv', index_col=['index'])
data = pd.concat([data, temp], axis=1)
del temp
gc.collect()

data.to_csv('data/merge_add_features.csv')


## Divid data
train = data.iloc[:train_test_boundary]
test = data.iloc[train_test_boundary:]

del data
gc.collect()

train.to_csv('data/train_add_features.csv', index=False)
test.to_csv('data/test_add_features.csv', index=False)


##
def explor_click_gap(data):
    temp = data.loc[data['click_gap'].isnull()]
    
    print('click_gap NaN : ', temp.shape)
    print(temp.head())
    print(temp.tail())
    
    print('ip count : \n', temp['ip'].value_counts())
    print('is attributed count : \n', temp['is_attributed'].sum())
    
    temp = data.loc[data['click_gap'] == 0, ['ip', 'is_attributed']]
    
    print('click_gap 0 : ', temp.shape)
    print(temp.head())
    print(temp.tail())
    
    print('ip count : \n', temp['ip'].value_counts())
    print('is attributed count : \n', temp['is_attributed'].sum())

explor_click_gap(train)
explor_click_gap(test)


#
print(train['click_gap'].max())         # 218,978

temp = train.loc[train['click_gap'].isnull()]
print('click_gap NaN : ', temp.shape)   # (277396, 26)
print(temp.head())
print(temp.tail())

print(temp['is_attributed'].sum())      # 70,584

def scatter_plot(df, feat):
    name = ['gap_' + feat, 'black_' + feat, 'rate_' + feat]
    
    g = sns.pairplot(df,
                     vars=name,
                     hue='is_attributed', 
                     palette="husl",
                     plot_kws={'alpha':0.1})
    
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(60)
    
    g.fig.set_size_inches(10,8)
    plt.savefig('graph/scatter_plot_'+feat+'_click_gap_NaN.png')
    plt.show()
    gc.collect()
    
for f in ['ip', 'app', 'device', 'os', 'channel', 'hour']:
    scatter_plot(temp, f)
    



## Extract a sample
import random

for n in [10000000, 20000000,30000000,40000000,50000000]:
    idx = random.sample(range(train_test_boundary),n)
    sample = train.iloc[idx]
    gc.collect()
    
    n = n / 1000000
    sample.to_csv('train_add_features_' + str(n) + 'm.csv', index=False)
    
    del sample
   
    
##
def scatter_plot(feat):
    temp = pd.read_csv('data/merge_is_attributed.csv', index_col=['index']) 
    name = ['gap_' + feat, 'black_' + feat, 'rate_' + feat]
    
    for n in name:
        x = pd.read_csv('data/merge_' + n + '.csv', index_col=['index'])
        temp = pd.concat([temp, x], axis=1)
        del x
        gc.collect()
    
    temp = temp.iloc[:184903890]
    gc.collect()
        
    g = sns.pairplot(temp,
                     vars=name,
                     hue='is_attributed', 
                     palette="husl",
                     plot_kws={'alpha':0.1})
    
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(60)
    
    g.fig.set_size_inches(10,8)
    plt.savefig('graph/scatter_plot_blck_'+feat+'.png')
    plt.show()
    gc.collect()

for f in ['ip', 'app', 'device', 'os', 'channel', 'hour']:
    scatter_plot(f)

##
temp = pd.read_csv('data/merge_is_attributed.csv', index_col=['index'])

x = pd.read_csv('data/merge_black_ip.csv', index_col=['index'])
temp = pd.concat([temp, x], axis=1)
del x

x = pd.read_csv('data/merge_click_gap.csv', index_col=['index'])
temp = pd.concat([temp, x], axis=1)
del x

temp = temp.iloc[:184903890]
gc.collect()

g = sns.pairplot(temp,
                 vars=['black_ip', 'click_gap'],
                 hue='is_attributed', 
                 palette="husl",
                 plot_kws={'alpha':0.1})

for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(60)

g.fig.set_size_inches(10,8)
plt.savefig('graph/scatter_plot_click_time.png')
plt.show()
gc.collect()

## check correlation
corr = train.corr(method='pearson')
corr = corr.round(2)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)

plt.figure(figsize=(17,17))
sns.heatmap(corr, vmin=-1, vmax=1,
            mask=mask, cmap=cmap, annot=True, linewidth=.5, cbar_kws={'shrink':.6})
plt.savefig('graph/heatmap3.png')
plt.show()
gc.collect()