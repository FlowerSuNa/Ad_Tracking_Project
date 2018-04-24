
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


## Create functions
def check_data(is_attributed):
    a1 = 0
    a09 = 0
    a07 = 0
    a05 = 0
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
        elif is_attributed[i] >= 0:
            a0 += 1
        else:
            a00 += 1
            
    print(a00,a0,a05,a07,a09,a1)
    
def outlier(is_attributed):
    for i in range(len(is_attributed)):
        if is_attributed[i] < 0:
            is_attributed[i] = 0
        if is_attributed[i] > 1:
            is_attributed[i] = 1
    return is_attributed


## Divid data
from sklearn.model_selection import train_test_split

feat = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop',
        'channel_attr_prop','tot_prop']
X_train, X_test, y_train, y_test = train_test_split(ad[feat], ad['is_attributed'])

print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))


## Make model using K-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_auc_score

for i in range(2,21):
    print("When n_neighbors is %d" %i)
    
    ## Use K-Nearest Neighbors
    reg = KNeighborsRegressor(n_neighbors=i)
    reg.fit(X_train,y_train)
    
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
print(is_attributed[1:100])

is_attributed = outlier(is_attributed)
submission['is_attributed'] = is_attributed
submission.to_csv('sample_submission_lr.csv', index=False)


## Make model using Ridge
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

## Use Ridge
ridge = Ridge(alpha=10)
ridge.fit(X_train,y_train)

## Evaluate a model
print("coefficient of determination : %.5f" % ridge.score(X_test,y_test))
print("AUC : %.5f" % roc_auc_score(y_test, ridge.predict(X_test)))


## Predict is_attributed
is_attributed = ridge.predict(ad_test[feat])
print(is_attributed[1:100])

check_data(is_attributed)
is_attributed = outlier(is_attributed)
check_data(is_attributed)

submission['is_attributed'] = is_attributed
submission.to_csv('sample_submission_ridge.csv', index=False)


## Make model using Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

for c in [0.01,0.1,1,10,100]:
    ## Use Logistic Regression
    logreg = LogisticRegression(C=c)
    logreg.fit(X_train,y_train)
    
    ## Evaluate a model
    print("When c is %d" %c)
    print("coefficient of determination : %.5f" % logreg.score(X_test,y_test))
    print("AUC : %.5f" % roc_auc_score(y_test, logreg.predict(X_test)))

## Predict is_attributed
is_attributed = logreg.predict(ad_test[feat])
print(is_attributed[1:100])
check_data(is_attributed)


## Make model using Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

## Use Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)

## Evaluate a model
print(tree.feature_importances_)
print("coefficient of determination : %.5f" % tree.score(X_test,y_test))
print("AUC : %.5f" % roc_auc_score(y_test, tree.predict_proba(X_test)[:,1]))

## Predict is_attributed
is_attributed = tree.predict_proba(ad_test[feat])[:,1]
print(is_attributed[1:100])

check_data(is_attributed)

submission['is_attributed'] = is_attributed
submission.to_csv('sample_submission_tree.csv', index=False)


## Make model using Geadient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

max_value = 0

for x in [3,5,10]:
    for y in [0.01,0.1,1]:
        for z in [50,70,100]:
            ## Use Gradient Boosting Classifier
            gbrt = GradientBoostingClassifier(max_depth=x, learning_rate=y, n_estimators=z)
            gbrt.fit(X_train,y_train)
            
            ## Evaluate a model
            roc = roc_auc_score(y_test, gbrt.predict_proba(X_test)[:,1])
            print("when max_depth=%d, learning_rate=%.2f, n_estimators=%d" % (x,y,z))
            print("feature : %s" % gbrt.feature_importances_)
            print("accuracy : %.5f" % gbrt.score(X_test,y_test))
            print("AUC : %.5f" % roc)
            
            if roc > max_value:
                max_value = roc
                ix = x
                iy = y
                iz = z

## Predict is_attributed
gbrt = GradientBoostingClassifier(max_depth=ix, learning_rate=iy, n_estimators=iz)
gbrt.fit(X_train,y_train)

is_attributed = gbrt.predict_proba(ad_test[feat])[:,1]
print(is_attributed[1:100])

check_data(is_attributed)

submission['is_attributed'] = is_attributed
submission.to_csv('sample_submission_gbrt.csv', index=False)


## Make model using Support Vector Machines
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

max_value = 0

for c in [0.1,1,10,100,1000]:
    for g in [0.1,1,10]:
        ## Use Decision Tree
        svm = SVC(C=c, gamma=g, probability=True)
        svm.fit(X_train,y_train)
        
        ## Evaluate a model
        roc = roc_auc_score(y_test, svm.predict_proba(X_test)[:,1])
        print("when C=%.2f, gamma=%.2f" % (c,g))
        print("accuracy : %.5f" % svm.score(X_test,y_test))
        print("AUC : %.5f" % roc)
        
        if roc > max_value:
            max_value = roc
            ic = c
            ig = g

## Predict is_attributed
svm = SVC(C=ic, gamma=ig, probability=True)
svm.fit(X_train,y_train)

is_attributed = svm.predict_proba(ad_test[feat])[:,1]
print(is_attributed[1:100])

check_data(is_attributed)

submission['is_attributed'] = is_attributed
submission.to_csv('sample_submission_svm.csv', index=False)