##### TalkingData AdTracking Fraud Detection Challenge
# 3. Sampling
[source code](03_Sampling.py)

<br>

---

## Import library

```python
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import gc

train_length = 184903890
```

<br>

## Extract a sample

```python
train = pd.read_csv('data/merge_add_features.csv')
index = train_length - 50000000
temp = train.iloc[index:]

for n in [40000000,30000000,20000000,10000000]:
    index = train_length - n
    sample = train.iloc[index:]
    print(sample.shape)
    gc.collect()

    m = n / 1000000
    sample.to_csv('data/train_add_features_' + str(m) + 'm.csv', index=False)

    del sample
```

<br>

---

## Draw dirstribution of sample

```python
def dist(a):
    df = pd.read_csv('data/train_add_features_20m.csv', usecols=[a, 'is_attributed'])

    g =  sns.FacetGrid(df, hue='is_attributed', size=7, palette='husl')
    g = g.map(sns.distplot, a, hist_kws={'alpha':0.2})

    plt.xticks(rotation=30, fontsize="small")
    plt.legend(loc='upper right').set_title('is_attributed')
    plt.savefig('graph/dist_' + a + '_20m.png', bbox_inches='tight')
    plt.show()
    gc.collect()
```
