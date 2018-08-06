
## 2. Preprocessing
## Import library
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import gc

train_test_boundary = 184903890


## Make features : gap, black and rate per (ip, app, device, os, channel, hour)
def merge_black(feat):
    df = pd.read_csv('data/merge.csv', usecols=[feat])    
    black = pd.read_csv('blacklist/' + feat + '_black.csv', usecols=[feat, 'gap','black_'+feat, 'rate'])
    
    df = df.merge(black, on=feat, how='left')
    df.rename(columns={'gap':'gap_'+feat}, inplace = True)
    df.rename(columns={'rate':'rate_'+feat}, inplace = True)
    gc.collect()
    
    df.to_csv('data/merge_' + feat + '.csv', index=False)

for feat in ['ip', 'app', 'device', 'os', 'channel', 'hour']:
    merge_black(feat)


## Make a feature : click_gap
data = pd.read_csv('data/merge.csv', usecols=['ip', 'click_time'])
data = data.reset_index(name='index')
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

data.loc[index, 'click_gap'] = np.nan
data['click_gap'] = data['click_gap'].astype('timedelta64[s]')
data.sort_values(by='index', ascending=True, inplace=True)

temp = data['click_gap']
temp.to_csv('data/merge_click_gap.csv', index=False)
  

## Concat data
data = pd.read_csv('data/merge.csv', usecols=['is_attributed'])

for feat in ['ip','app','device','os','channel','hour']:
    temp = pd.read_csv('data/merge_ip.csv')
    data = pd.concat([data, temp], axis=1)
    del temp
    gc.collect()

temp = pd.read_csv('data/merge_click_gap.csv')
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

del train
del test
gc.collect()


## Draw distribution
def dist(a):
    df = pd.read_csv('data/train_add_features.csv', usecols=[a, 'is_attributed'])
    
    g =  sns.FacetGrid(df, hue='is_attributed', height=7, palette='husl')
    g = g.map(sns.distplot, a, hist_kws={'alpha':0.2})
    
    plt.xticks(rotation=30, fontsize="small")
    plt.legend(loc='upper right').set_title('is_attributed')
    plt.savefig('graph/dist_' + a + '.png', bbox_inches='tight')
    plt.show()
    gc.collect()
    
dist('gap_ip')
dist('rate_ip')

dist('gap_app')
dist('rate_app')

dist('gap_device')
dist('rate_device')

dist('gap_os')
dist('rate_os')

dist('gap_channel')
dist('rate_channel')

dist('gap_hour')
dist('rate_hour')

dist('click_gap')


##
def scatter(feat):
    x = pd.read_csv('data/train_add_features.csv', usecols=feat+['is_attributed'])
    
    g = sns.pairplot(x,
                     vars=feat,
                     hue='is_attributed', 
                     palette="husl",
                     plot_kws={'alpha':0.1})
    
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(60)
    
    g.fig.set_size_inches(20,18)
    plt.savefig('graph/scatter.png', bbox_inches='tight')
    plt.show()
    gc.collect()
    
scatter(['gap_ip', 'gap_app', 'gap_device', 'gap_os', 'gap_channel'])
    


##
def bar(x):
    df = pd.read_csv('data/train_add_features.csv', usecols=[x, 'is_attributed'])
    
    sns.set(rc={'figure.figsize':(10,5)})
    
    temp = df.loc[df['is_attributed'] == 0, x]
    plt.subplot(1,2,1)
    plt.title('Not Downloaded')
    sns.countplot(x, data=temp, linewidth=0, palette='husl')
    
    temp = df.loc[df['is_attributed'] == 1, x]
    plt.subplot(1,2,2)
    plt.title('Downloaded')
    sns.countplot(x, data=temp, linewidth=0, palette='husl')
    
    plt.savefig('graph/bar_' + x + '.png', bbox_inches='tight')
    plt.show()
    gc.collect()

bar('black_ip')
bar('black_app')
bar('black_device')
bar('black_os')
bar('black_channel')
bar('black_hour')



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


temp = train.loc[train['click_gap'] == 0]
print(temp.shape)       # (48174407, 26)
print(temp['is_attributed'].sum()) # 28,830

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
    
    
train.loc[train['click_gap'].notnull()]
    
    
name = ['gap_ip', 'click_gap']

g = sns.pairplot(train,
                 vars=name,
                 hue='is_attributed', 
                 palette="husl",
                 plot_kws={'alpha':0.1})

for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(60)

g.fig.set_size_inches(10,8)
plt.savefig('graph/scatter_plot_click_gap_gap_ip.png')
plt.show()
gc.collect()







name = ['black_ip', 'click_gap']

g = sns.pairplot(train,
                 vars=name,
                 hue='is_attributed', 
                 palette="husl",
                 plot_kws={'alpha':0.1})

for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(60)

g.fig.set_size_inches(10,8)
plt.savefig('graph/scatter_plot_click_gap_black_ip.png')
plt.show()
gc.collect()

    


## Extract a sample
import random

for n in [10000000, 20000000,30000000,40000000,50000000]:
    idx = random.sample(range(len(train)),n)
    sample = train.iloc[idx]
    gc.collect()
    
    n = n / 1000000
    sample.to_csv('data/train_add_features_' + str(n) + 'm.csv', index=False)
    
    del sample
    

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