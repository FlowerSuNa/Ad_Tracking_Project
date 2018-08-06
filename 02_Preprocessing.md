##### TalkingData AdTracking Fraud Detection Challenge
# 2. Preprocessing
[source code](02_Preprocessing.py)

<br>

---

## Import library

```python
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import gc

train_test_boundary = 184903890
```

<br>

## Make features : gap, black and rate per (ip, app, device, os, channel, hour)

```python
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
```

```python
for feat in ['ip', 'app', 'device', 'os', 'channel', 'hour']:
    merge_black(feat)
```

<br>

## Draw scatter plots

```python

```
