
## 3.Sampling
## Import library
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import gc



## Extract a sample
import random

train = pd.read_csv('data/merge_add_features.csv')

for n in [10000000, 20000000,30000000,40000000,50000000]:
    idx = random.sample(range(len(train)),n)
    sample = train.iloc[idx]
    gc.collect()
    
    n = n / 1000000
    sample.to_csv('data/train_add_features_' + str(n) + 'm.csv', index=False)
    
    del sample
    
    
## Draw distribution
def dist(a):
    df = pd.read_csv('data/train_add_features_20m.csv', usecols=[a, 'is_attributed'])
    
    g =  sns.FacetGrid(df, hue='is_attributed', size=7, palette='husl')
    g = g.map(sns.distplot, a, hist_kws={'alpha':0.2})
    
    plt.xticks(rotation=30, fontsize="small")
    plt.legend(loc='upper right').set_title('is_attributed')
    plt.savefig('graph/dist_' + a + '_20m.png', bbox_inches='tight')
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
    x = pd.read_csv('data/train_add_features_20m.csv', usecols=feat+['is_attributed'])
    
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
    df = pd.read_csv('data/train_add_features_20m.csv', usecols=[x, 'is_attributed'])
    
    sns.set(rc={'figure.figsize':(10,5)})
    
    temp = df.loc[df['is_attributed'] == 0]
    plt.subplot(1,2,1)
    plt.title('Not Downloaded')
    sns.countplot(x, data=temp, linewidth=0, palette='husl')
    
    temp = df.loc[df['is_attributed'] == 1]
    plt.subplot(1,2,2)
    plt.title('Downloaded')
    sns.countplot(x, data=temp, linewidth=0, palette='husl')
    
    plt.savefig('graph/bar_' + x + '_20m.png', bbox_inches='tight')
    plt.show()
    gc.collect()

bar('black_ip')
bar('black_app')
bar('black_device')
bar('black_os')
bar('black_channel')
bar('black_hour')


##
feat = ['black_ip', 'black_app', 'black_device','black_os','black_channel','is_attributed']
train = pd.read_csv('data/train_add_features_20m.csv', usecols=feat)

train.loc[train[feat].sum(axis=1) == 5, 'is_attributed'].sum() # 5416 / 19249912

train = pd.read_csv('data/train_add_features_20m.csv', usecols=['click_gap', 'is_attributed'])
print(train.shape)  # (20000000, 2)
print(train['is_attributed'].sum()) # 49,082

temp = train.loc[train['click_gap'].isnull(), 'is_attributed']
print(temp.shape)   # (29755, )
print(temp.sum())   # 7,532

temp = train.loc[train['click_gap'] == 0, 'is_attributed']
print(temp.shape)   # (5210168, )
print(temp.sum())   # 3,127

temp = train.loc[train['click_gap'] > 50, 'is_attributed']
print(temp.shape)   # (2983405,)
print(temp.sum())   # 18,819

temp = train.loc[train['click_gap'] > 100, 'is_attributed']
print(temp.shape)   # (1997292, )
print(temp.sum())   # 15,501

temp = train.loc[train['click_gap'] > 200, 'is_attributed']
print(temp.shape)   # (1217415, )
print(temp.sum())   # 12,341


train = pd.read_csv('data/train_add_features_50m.csv')
train.loc[train['click_gap'].isnull(), 'click_gap'] = -1
train.to_csv('data/train_add_features_50m.csv', index=False)
del train

