
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
    return str(a00) + ' ' + str(a0) + ' ' + str(a03) + ' ' + str(a05) + ' ' + str(a07) + ' ' + str(a09) + ' ' + str(a1)

    
def examine_outlier(is_attributed):
    r = check_data(is_attributed)
    
    if (is_attributed.min() < 0) | (is_attributed.max() > 1):
        for i in range(len(is_attributed)):
            if is_attributed[i] < 0:
                is_attributed[i] = 0
            if is_attributed[i] > 1:
                is_attributed[i] = 1
        r = check_data(is_attributed)
            
    return is_attributed, r 


## Import library and data
import pandas as pd

ad = pd.read_csv('ad_modify2_10m.csv')
print(ad.columns)

ad_test = pd.read_csv('adtest_modify_all.csv')
print(ad_test.columns)

submission = pd.read_csv('sample_submission.csv')
print(submission.columns)


## Make a result DataFrame
from pandas import DataFrame

i = 0
colnames = ['name','param', 'auc','train','test']
result = pd.DataFrame(columns=colnames)


## Check correlation
feat1 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop',
         'channel_attr_prop','tot_attr_prop']

feat2 = ['ip_time_prop','ip_app_prop','ip_channel_prop','time_app_prop',
         'time_channel_prop','tot_vv_prop']

feat3 = feat1 + feat2

print(ad[feat3 + ['is_attributed']].corr(method='pearson'))


## Select features
feat = feat3


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
p, train = examine_outlier(p)

## Evaluate the model
score = lr.score(X_test,y_test)
auc = roc_auc_score(y_test, lr.predict(X_test))

print("coefficient : %s" % lr.coef_)
print("intercept : %s" % lr.intercept_)
print("coefficient of determination : %.5f" % score)
print("AUC : %.5f" % auc)

## Predict target variable
is_attributed = lr.predict(ad_test[feat])
is_attributed, test = examine_outlier(is_attributed)

## Save result
r = pd.DataFrame({'name':'lr',
                  'param':'null',
                  'auc':auc,
                  'train':train,
                  'test':test}, columns=colnames, index=[i])
result = pd.concat([result, r], ignore_index=True)
i+=1    

submission['is_attributed'] = is_attributed
submission.to_csv('10m_submission_lr.csv', index=False)


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
    p, train = examine_outlier(p)
    
    ## Evaluate the model
    score = ridge.score(X_test,y_test)
    auc = roc_auc_score(y_test, p)
    
    print("coefficient of determination : %.5f" % score)
    print("AUC : %.5f" % auc)

    ## Predict target variable
    is_attributed = ridge.predict(ad_test[feat])
    is_attributed, test = examine_outlier(is_attributed)

    ## Save result
    r = pd.DataFrame({'name':'ridge',
                      'param':str(a),
                      'auc':auc,
                      'train':train,
                      'test':test}, columns=colnames, index=[i])
    result = pd.concat([result, r], ignore_index=True)
    i+=1     


## Make a model using Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

print("y_test : ")
print(y_test.value_counts())

for c in [1]:
    print("When C=%.2f :" %c)
    
    ## Train a model
    logreg = LogisticRegression(C=c)
    logreg.fit(X_train,y_train)

    ## predict is_attributed
    p = logreg.predict_proba(X_test)[:,1]
    p, train = examine_outlier(p)
        
    ## Evaluate the model
    score = logreg.score(X_test,y_test)
    auc = roc_auc_score(y_test, p)
    
    print("coefficient of determination : %.5f" % score)
    print("AUC : %.5f" % auc)

    ## Predict target variable
    is_attributed = logreg.predict_proba(ad_test[feat])[:,1]
    is_attributed, test = examine_outlier(is_attributed)
    
    ## Save result
    r = pd.DataFrame({'name':'logreg',
                      'param':str(c),
                      'auc':auc,
                      'train':train,
                      'test':test}, columns=colnames, index=[i])
    result = pd.concat([result, r], ignore_index=True)
    i+=1    
        
    submission['is_attributed'] = is_attributed
    submission.to_csv('10m_submission_logreg_'+str(c)+'.csv', index=False)
    

## Save resultset
result.to_csv('result.csv', index=False)