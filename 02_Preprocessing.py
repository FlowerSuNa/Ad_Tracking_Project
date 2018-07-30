
## 2. Preprocessing
## Import library
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import gc


## Load data
data = pd.read_csv("merge.csv", parse_dates=['click_time'])
ip = pd.read_csv("blacklist/ip_download.csv")
app = pd.read_csv("blacklist/app_download.csv")
device = pd.read_csv("blacklist/device_download.csv")
os = pd.read_csv("blacklist/os_download.csv")
gc.collect()

print(data.head())
print(data.tail())


##
def merge_black(df, black, feat):
    temp = black[[feat, 'gap','black_'+feat]]
    df = df.merge(temp, on=feat, how='left')
    df.rename(columns={'gap':'gap_'+feat}, inplace = True)
    gc.collect()
    
    print(df.head())
    return df
    
data = merge_black(data, ip, 'ip')
data = merge_black(data, app, 'app')
data = merge_black(data, device, 'device')
data = merge_black(data, os, 'os')
data.to_csv('merge_gap_black.csv', index=False)


##
def scatter_plot(feat, file_name):
    temp = data[feat].loc[data['click_id'].isnull()]
    
    plt.figure(figsize=(20,20))
    sns.pairplot(temp, 
                 hue='is_attributed', 
                 palette="husl",
                 plot_kws={'alpha':0.1})
    plt.xticks(rotation=90, fontsize="small")
    plt.savefig('graph/'+file_name+'.png')
    plt.show()
    gc.collect()


feat = ['is_attributed', 'gap_ip', 'black_ip']
scatter_plot(feat, 'scatter_plot_gap_black_ip')



feat = ['is_attributed', 'gap_ip', 'black_ip']
temp = data[feat].loc[data['click_id'].isnull()]

plt.figure(figsize=(20,20))
sns.pairplot(temp, 
             hue='is_attributed', 
             palette="husl",
             plot_kws={'alpha':0.1})
plt.savefig('graph/scatter_plot_gap_black_ip.png')
plt.show()
gc.collect()


##

feat = ['is_attributed', 'gap_app', 'black_app']
plt.figure(figsize=(20,20))

sns.pairplot(data[feat].loc[data['click_id'].isnull()], 
                  hue='is_attributed', 
                  palette="husl", 
                  diag_kind="kde",
                  plot_kws={'alpha':0.1})

plt.savefig('graph/scatter_plot_gap_black_app.png')
plt.show()
gc.collect()
    
    
##
feat = ['is_attributed', 'gap_ip', 'black_ip', 'gap_app', 'black_app', 'gap_device', 'black_device', 'gap_os', 'black_os']
plt.figure(figsize=(20,20))
sns.pairplot(data[feat].loc[data['click_id'].isnull()], palette="husl", alpha=.2)
plt.savefig('graph/scatter_plot_gap_black.png')
plt.show()
gc.collect()    
    


# Make a derived variable : click_gap
train = pd.read_csv('train.csv')
for ip in train.ip.value_counts().index:
    temp = train.loc[train['ip'] == ip, 'click_time']
    temp.sort_values(ascending=False, inplace=True)
    
    temp['click_gap'] = np.nan
    temp[1:].loc['click_gap'] = temp[:-1].loc['click_time']
    temp['click_gap'] = temp['click_gap'] - temp['click_time']

temp = train.loc[train['ip'] == 10, 'click_time'].reset_index()
temp.sort_values(ascending=True, by='click_time', inplace=True)

temp['click_gap'] = np.nan
temp['click_gap'] = temp['click_time']
temp['click_gap'].iloc[1:] = temp['click_time']


temp[1:].loc['click_gap'] = temp[0:len(temp) - 1].loc['click_time']
temp['click_gap'] = temp['click_gap'] - temp['click_time']
    
    
a = train['click_time'].iloc[1] - train['click_time'].iloc[0]





