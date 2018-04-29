
## 4. Target Variable Prediction - Decision Tree, Random Forest, Gradient Boosting
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

ad_test = pd.read_csv('adtest_modify_all.csv')
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


## Make a model using Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

print("y_test : ")
print(y_test.value_counts())

for d in [3,5,7,10]:
    print("When max_depth=%d :" %d)
        
    ## Train a model
    tree = DecisionTreeClassifier(max_depth=d, random_state=1)
    tree.fit(X_train,y_train)

    ## predict is_attributed
    p = tree.predict_proba(X_test)[:,1]
    p = examine_outlier(p)

    ## Evaluate the model
    print("feature : %s" % tree.feature_importances_)
    print("coefficient of determination : %.5f" % tree.score(X_test,y_test))
    print("AUC : %.5f" % roc_auc_score(y_test, p))

    ## Predict target variable
    is_attributed = tree.predict_proba(ad_test[feat])[:,1]
    is_attributed = examine_outlier(is_attributed)

    submission['is_attributed'] = is_attributed
    submission.to_csv('10m_submission_tree_'+str(d)+'.csv', index=False)


## Make a model using Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

print("y_test : ")
print(y_test.value_counts())
    

for d in [3,5,7]:
    for e in [30,50,70,100]:
        for f in [1,2,3,4]:
            print("When max_depth=%d, n_estimators=%d, max_features=%d :" %(d,e,f))
            
            ## Train a model
            forest = RandomForestClassifier(max_depth=d, n_estimators=e, max_features=f, random_state=1)
            forest.fit(X_train,y_train)
    
            ## predict is_attributed
            p = forest.predict_proba(X_test)[:,1]
            p = examine_outlier(p)
            
            ## Evaluate the model
            print("feature : %s" % forest.feature_importances_)
            print("coefficient of determination : %.5f" % forest.score(X_test,y_test))
            print("AUC : %.5f" % roc_auc_score(y_test, p))
        
            ## Predict target variable
            is_attributed = forest.predict_proba(ad_test[feat])[:,1]
            is_attributed = examine_outlier(is_attributed)
    
            # submission['is_attributed'] = is_attributed
            # submission.to_csv('sample_submission_forest_'+str(d)+'_'+str(e)+'.csv', index=False)


## Make a model using Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

print("y_test : ")
print(y_test.value_counts())
    
for d in [3,4,5]:
    for e in [30,50]:
        for l in [0.01,0.1,1,10]:
            print("when max_depth=%d, n_estimators=%d, learning_rate=%.2f : " %(d,e,l))
    
            ## Train a model
            gbrt = GradientBoostingClassifier(max_depth=d, n_estimators=e, learning_rate=l)
            gbrt.fit(X_train,y_train)

            ## predict is_attributed
            p = gbrt.predict_proba(X_test)[:,1]
            p = examine_outlier(p)
    
            ## Evaluate the model
            print("feature : %s" % gbrt.feature_importances_)
            print("coefficient of determination : %.5f" % gbrt.score(X_test,y_test))
            print("AUC : %.5f" % roc_auc_score(y_test, p))
    
            ## Predict target variable
            is_attributed = gbrt.predict_proba(ad_test[feat])[:,1]
            is_attributed = examine_outlier(is_attributed)

            submission['is_attributed'] = is_attributed
            submission.to_csv('sample_submission_gbrt_'+str(d)+'_'+str(l) +'_'+str(e)+'.csv', index=False)