
## 3. Data Analytics
## Import library and data
import pandas as pd
import numpy as np

ad = pd.read_csv('ad_dataset_modify.csv')
print(ad.columns)

ad_test = pd.read_csv('adtest_dataset_modify.csv')
print(ad_test.columns)

submission = pd.read_csv('sample_submission.csv')
print(submission.columns)


## Check correlation
print(ad[['app', 'device', 'os', 'channel',
          'ip_attr_prop', 'app_attr_prop', 'device_attr_prop', 'os_attr_prop', 
          'channel_attr_prop', 'hour_attr_prop', 'tot_prop', 'is_attributed']].corr(method='pearson'))


## Divid data
from sklearn.model_selection import train_test_split

feat = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop',
        'channel_attr_prop','hour_attr_prop','tot_prop']
X_train, X_test, y_train, y_test = train_test_split(ad[feat], ad['is_attributed'])

print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))


## Make model using K-Neighbors Regressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_auc_score

for i in range(2,21):
    print("When n_neighbors is %d" %i)
    
    ## Use Decision Tree
    reg = KNeighborsRegressor(n_neighbors=i)
    reg.fit(X_train, y_train)
    
    ## Evaluate a model  
    print("coefficient of determination : %.5f" % reg.score(X_test,y_test))
    print("AUC : %.5f" % roc_auc_score(y_test, reg.predict(X_test)))


## Make model using Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

## Use Linear Regression
lr = LinearRegression()
lr.fit(X_train,y_train)

## Evaluate a model
print("coefficient : %s" % lr.coef_)
print("intercept : %s" % lr.intercept_)
print("coefficient of determination : %.5f" % lr.score(X_test,y_test))
print("AUC : %.5f" % roc_auc_score(y_test, lr.predict(X_test)))


## Predict is_attributed
is_attributed = lr.predict(ad_test[feat])
is_attributed

from collections import Counter
print(Counter(is_attributed))

submission['is_attributed'] = is_attributed
submission.to_csv('sample_submission1.csv', index=False)