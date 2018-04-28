
## 4. Target Variable Prediction - Linear Regression, Ridge, Logistic Regression
## Create functions
def check_data(is_attributed):
    a1 = 0
    a09 = 0
    a07 = 0
    a05 = 0
    a03 = 0
    a0 = 0
    a00 = 0
    
    for i in range(len(is_attributed)):
        if is_attributed[i] > 1:
            a1 += 1
        elif is_attributed[i] > 0.9:
            a09 += 1
        elif is_attributed[i] > 0.7:
            a07 += 1
        elif is_attributed[i] > 0.5:
            a05 += 1
        elif is_attributed[i] > 0.3:
            a03 += 1
        elif is_attributed[i] >= 0:
            a0 += 1
        else:
            a00 += 1
            
    print(a00,a0,a03,a05,a07,a09,a1)

    
def examine_outlier(is_attributed):
    check_data(is_attributed)
    
    if (is_attributed.min() < 0) | (is_attributed.max() > 1):
        for i in range(len(is_attributed)):
            if is_attributed[i] < 0:
                is_attributed[i] = 0
            if is_attributed[i] > 1:
                is_attributed[i] = 1
        check_data(is_attributed)
            
    return is_attributed


## Import library and data
import pandas as pd
import numpy as np

ad = pd.read_csv('ad_modify2_10m.csv')
print(ad.columns)

ad_test = pd.read_csv('adtest_modify_10m.csv')
print(ad_test.columns)

submission = pd.read_csv('sample_submission.csv')
print(submission.columns)


## Check correlation
feat = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop',
        'channel_attr_prop','tot_attr_prop',
        'ip_time_prop','ip_app_prop','ip_channel_prop','time_app_prop',
        'time_channel_prop','tot_vv_prop']

print(ad[feat + ['is_attributed']].corr(method='pearson'))


## Divid data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(ad[feat], ad['is_attributed'], random_state=1)

print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))


## Make a model using Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

print("y_test : ")
print(y_test.value_counts())

## Train a model
lr = LinearRegression()
lr.fit(X_train,y_train)

## predict is_attributed
p = lr.predict(X_test)
p = examine_outlier(p)

## Evaluate the model
print("coefficient : %s" % lr.coef_)
print("intercept : %s" % lr.intercept_)
print("coefficient of determination : %.5f" % lr.score(X_test,y_test))
print("AUC : %.5f" % roc_auc_score(y_test, lr.predict(X_test)))

## Predict target variable
is_attributed = lr.predict(ad_test[feat])
is_attributed = examine_outlier(is_attributed)

submission['is_attributed'] = is_attributed
submission.to_csv('sample_submission_lr.csv', index=False)


## Make a model using Ridge
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

print("y_test : ")
print(y_test.value_counts())

for a in [0.1,1,10]:
    print("When alpha=%.1f :" %a)
    
    ## Train a model
    ridge = Ridge(alpha=a)
    ridge.fit(X_train,y_train)
    
    ## predict is_attributed
    p = ridge.predict(X_test)
    p = examine_outlier(p)
    
    ## Evaluate the model
    print("coefficient of determination : %.5f" % ridge.score(X_test,y_test))
    print("AUC : %.5f" % roc_auc_score(y_test, p))

    ## Predict target variable
    is_attributed = ridge.predict(ad_test[feat])
    is_attributed = examine_outlier(is_attributed)


## Make a model using Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

print("y_test : ")
print(y_test.value_counts())

for c in [0.01,0.1,1,10,100]:
    print("When C=%.2f :" %c)
    
    ## Train a model
    logreg = LogisticRegression(C=c)
    logreg.fit(X_train,y_train)

    ## predict is_attributed
    p = logreg.predict_proba(X_test)[:,1]
    p = examine_outlier(p)
        
    ## Evaluate the model
    print("coefficient of determination : %.5f" % logreg.score(X_test,y_test))
    print("AUC : %.5f" % roc_auc_score(y_test, p))

    ## Predict target variable
    is_attributed = logreg.predict_proba(ad_test[feat])[:,1]
    is_attributed = examine_outlier(is_attributed)
        
    submission['is_attributed'] = is_attributed
    submission.to_csv('sample_submission_logreg_'+str(c)+'.csv', index=False)