
## 4. Target Variable Prediction - K-Nearest Neighbors, Support Vector Machines
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
feat = ['ip_cnt', 'ip_attr', 'ip_attr_prop', 'app_cnt', 'app_attr',
       'app_attr_prop', 'device_cnt', 'device_attr', 'device_attr_prop',
       'os_cnt', 'os_attr', 'os_attr_prop', 'channel_cnt', 'channel_attr',
       'channel_attr_prop', 'hour_cnt', 'hour_attr', 'hour_attr_prop',
       'tot_prop', 'ip_attr_prop2', 'app_attr_prop2', 'device_attr_prop2',
       'os_attr_prop2', 'channel_attr_prop2', 'hour_attr_prop2']

print(ad[feat + ['is_attributed']].corr(method='pearson'))


## Divid data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(ad[feat], ad['is_attributed'], random_state=1)

print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))


## Make a model using K-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_auc_score

print("y_test : ")
print(y_test.value_counts())

for i in range(2,21):
    print("When n_neighbors=%d :" %i)

    ## Train a model
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train,y_train)

    ## predict is_attributed
    p = knn.predict(X_test)
    p = examine_outlier(p)

    ## Evaluate the model  
    print("coefficient of determination : %.5f" % knn.score(X_test,y_test))
    print("AUC : %.5f" % roc_auc_score(y_test,p))

    ## Predict target variable
    is_attributed = knn.predict(ad_test[feat])
    is_attributed = examine_outlier(is_attributed)


## Make a model using Support Vector Machines
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

for c in [0.1,1,10,100,1000]:
    for g in [0.1,1,10]:
        print("when C=%.1f , gamma=%.1f :" % (c,g))
    
        ## Train a model
        svm = SVC(C=c, gamma=g, probability=True)
        svm.fit(X_train,y_train)
    
        ## predict is_attributed
        p = svm.predict_proba(X_test)[:,1]
        p = examine_outlier(p)
    
        ## Evaluate the model
        print("coefficient of determination : %.5f" % svm.score(X_test,y_test))
        print("AUC : %.5f" % roc_auc_score(y_test, p))
    
        ## Predict target variable
        is_attributed = svm.predict_proba(ad_test[feat])[:,1]
        is_attributed = examine_outlier(is_attributed)

        submission['is_attributed'] = is_attributed
        submission.to_csv('sample_submission_svm_'+str(c)+'_'+str(g)+'.csv', index=False)