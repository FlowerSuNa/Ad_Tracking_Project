
## 2. Preprocessing
## Import library
import pandas as pd
import numpy as np
import gc



# Merge train data and test data
del train['attributed_time']

test['is_attributed'] = 0




# Make a derived variable : click_gap

for ip in train.ip.value_counts().index:
    temp = train
a = train['click_time'].iloc[1] - train['click_time'].iloc[0]





## Check click count, download count, download rate per hour
ad['hour'] = np.nan
ad['hour'] = ad['click_time'].dt.hour
barplot('hour', 'is_attributed', ad)