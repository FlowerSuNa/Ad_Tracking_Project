##### TalkingData AdTracking Fraud Detection Challenge
# 3. Sampling
[source code](03_Sampling.py) <br>

Extract a sample and draw graphs.

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

## Extract samples

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

> Extract the most recent data because it is time series data.

<br>

---

> Draw graphs to see if the sample data and the all data distribution are similar. <br>
> They are similar.

<br>

[View](03_sample_graphs.md)

<br>

---

[Contents](README.md) <br>
[2. Preprocessing](02_Preprocessing.md) <br>
[4. Modeling](04_Modeling.md)
