##### TalkingData AdTracking Fraud Detection Challenge
# sample graphs

Draw graphs to see if the sample data and the all data distribution are similar.

<br>

---

## Draw distribution of sample

* gap_ip

![png](graph/dist_gap_ip_20m.png)

<br>

* rate_ip

![png](graph/dist_rate_ip_20m.png)

<br>

* gap_app
![png](graph/dist_gap_app_20m.png)

<br>

* rate_app

![png](graph/dist_rate_app_20m.png)

<br>

* gap_device

![png](graph/dist_gap_device_20m.png)

<br>

* rate_device

![png](graph/dist_rate_device_20m.png)

<br>

* gap_os

![png](graph/dist_gap_os_20m.png)

<br>

* rate_os

![png](graph/dist_rate_os_20m.png)

<br>

* gap_channel

![png](graph/dist_gap_channel_20m.png)

<br>

* rate_channel

![png](graph/dist_rate_channel_20m.png)

<br>

* gap_hour

![png](graph/dist_gap_hour_20m.png)

<br>

* rate_hour

![png](graph/dist_rate_hour_20m.png)

<br>

## Draw bar graphs of sample

* black_ip

![png](graph/bar_black_ip_20m.png)

<br>

* black_app

![png](graph/bar_black_app_20m.png)

<br>

* black_device
![png](graph/bar_black_device_20m.png)

<br>

* black_os
![png](graph/bar_black_os_20m.png)

<br>

* black_channel

![png](graph/bar_black_channel_20m.png)

<br>

* black_hour

![png](graph/bar_black_hour_20m.png)

<br>

## Draw a bar graph of feature 'click_gap'

```python
train = pd.read_csv('data/train_add_features_20m.csv', usecols=['click_gap', 'is_attributed'])

sns.set(rc={'figure.figsize':(15,12)})

temp = train.loc[train['is_attributed'] == 0]
plt.subplot(2,1,1)
plt.title('Not Downloaded')
sns.countplot('click_gap', data=temp, linewidth=0)
plt.xlim((-1,20))

temp = train.loc[train['is_attributed'] == 1]
plt.subplot(2,1,2)
plt.title('Downloaded')
sns.countplot('click_gap', data=temp, linewidth=0)
plt.xlim((-1,20))

plt.savefig('graph/bar_click_gap_20m.png', bbox_inches='tight')
plt.show()
gc.collect()
```

![png](graph/bar_click_gap_20m.png)

<br>

## Check correlation

```python
train = pd.read_csv('data/train_add_features_20m.csv')
corr = train.corr(method='pearson')
corr = corr.round(2)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.set(rc={'figure.figsize':(20,18)})
sns.heatmap(corr, vmin=-1, vmax=1,
            mask=mask, cmap=cmap, annot=True, linewidth=.5, cbar_kws={'shrink':.6})
plt.savefig('graph/heatmap_20m.png', bbox_inches='tight')
plt.show()
gc.collect()
```

![png](graph/heatmap_20m.png)

---

[Contents](README.md) <br>
[3. Sampling](03_Sampling.md) <br>
[4. Modeling](04_Modeling.md)
