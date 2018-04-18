
## 2. Data Analysis
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
          'ip_cnt', 'ip_attr', 'ip_attr_prop', 'app_cnt', 'app_attr', 'app_attr_prop', 
          'device_cnt', 'device_attr', 'device_attr_prop', 'os_cnt', 'os_attr', 'os_attr_prop', 
          'channel_cnt', 'channel_attr', 'channel_attr_prop', 'click_hour', 'is_attributed']].corr(method='pearson'))


## Make model using Decision Tree
## Divid data
from sklearn.model_selection import train_test_split

feat = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop','channel_attr_prop']
X_train, X_test, y_train, y_test = train_test_split(ad[feat], ad['is_attributed'], random_state=0)

print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))

## Use Decision Tree
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=15, random_state=0)
tree.fit(X_train, y_train)

## Evaluate a model
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,tree.predict(X_test[feat])))
print(tree.score(X_train,y_train))
print(tree.score(X_test,y_test))
print(tree.feature_importances_)


## Predict is_attributed
is_attributed = tree.predict(ad_test[feat])
is_attributed

from collections import Counter
print(Counter(is_attributed))

submission['is_attributed'] = is_attributed
submission.to_csv('sample_submission1.csv', index=False)


## Make model using Forest Classifier
## Use Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=0)
forest.fit(X_train, y_train)

## Evaluate a model
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,tree.predict(X_test[feat])))
print(forest.score(X_train, y_train))
print(forest.score(X_test, y_test))
print(forest.feature_importances_)
