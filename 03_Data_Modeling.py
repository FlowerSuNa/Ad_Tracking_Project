
## 3. Data Modeling
## import library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

## proclaim functions
def extract_variables(data, var):
    var1 = []
    for v in var:
        if abs(data[[v,'is_attributed']].corr(method='pearson').loc[v,'is_attributed']) >=0.3:
            var1.append(v)
        elif abs(data[[v,'is_attributed']].corr(method='spearman').loc[v,'is_attributed']) >= 0.2:
            var1.append(v)
    print(var1)
    return var1


def divid_data(data, var):
    X_train, X_test, y_train, y_test = train_test_split(ad[var3], ad['is_attributed'], random_state=1)

    print("X_train : " + str(X_train.shape))
    print("X_test : " + str(X_test.shape))
    print("y_train : " + str(y_train.shape))
    print("y_test : " + str(y_test.shape))
    
    return X_train, X_test, y_train, y_test


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


## proclaim variable
var = ['ip','app','device','os','channel','click_hour']
var1 = ['ip_cnt','app_cnt','device_cnt','os_cnt','channel_cnt','hour_cnt']
var2 = ['ip_attr','app_attr','device_attr','os_attr','channel_attr','hour_attr']
var3 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop','channel_attr_prop',
        'hour_attr_prop','tot_prop']


## Import data
ad = pd.read_csv('ad_modify_10m.csv')
ad.drop(var, axis=1, inplace=True)
print(ad.columns)

var4 = extract_variables(ad, var1+var2+var3)


## Make model using K-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_auc_score

for v in [var3,var4]:
    print("variable : %s" %v)
    X_train, X_test, y_train, y_test = divid_data(ad, v)

    for i in range(2,21):
        print("When n_neighbors : %d ," %i)
    
        ## Train model Using K-Nearest Neighbors
        reg = KNeighborsRegressor(n_neighbors=i)
        reg.fit(X_train,y_train)
    
        ## Evaluate a model  
        print("coefficient of determination : %.5f" % reg.score(X_test,y_test))
        print("AUC : %.5f" % roc_auc_score(y_test, reg.predict(X_test)))


## Make model using Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

for v in [var3,var4]:
    print("variable : %s" %v)
    X_train, X_test, y_train, y_test = divid_data(ad, v)
    print("y_test : ")
    print(y_test.value_counts())

    ## Train model Using Linear Regression
    lr = LinearRegression()
    lr.fit(X_train,y_train)

    ## predict is_attributed
    p = lr.predict(X_test)
    p = examine_outlier(p)
    
    ## Evaluate a model
    print("coefficient of determination : %.5f" % lr.score(X_test,y_test))
    print("AUC : %.5f" % roc_auc_score(y_test, p))


## Make model using Ridge
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

for v in [var3,var4]:
    print("variable : %s" %v)
    X_train, X_test, y_train, y_test = divid_data(ad, v)
    print("y_test : ")
    print(y_test.value_counts())
    
    for a in [0.1,1,10]:
        print("When alpha : %.1f ," %a)
        
        ## Train model Using Ridge
        ridge = Ridge(alpha=a)
        ridge.fit(X_train,y_train)
        
        ## predict is_attributed
        p = ridge.predict(X_test)
        p = examine_outlier(p)
        
        ## Evaluate a model
        print("coefficient of determination : %.5f" % ridge.score(X_test,y_test))
        print("AUC : %.5f" % roc_auc_score(y_test, p))


## Make model using Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

for v in [var3,var4]:
    print("variable : %s" %v)
    X_train, X_test, y_train, y_test = divid_data(ad, v)
    print("y_test : ")
    print(y_test.value_counts())

    for c in [0.01,0.1,1,10,100]:
        print("When C : %.2f ," %c)
        
        ## Train model Using Logistic Regression
        logreg = LogisticRegression(C=c)
        logreg.fit(X_train,y_train)
   
        ## predict is_attributed
        p = logreg.predict_proba(X_test)[:,1]
        p = examine_outlier(p)
        
        ## Evaluate a model
        print("coefficient of determination : %.5f" % logreg.score(X_test,y_test))
        print("AUC : %.5f" % roc_auc_score(y_test, p))


## Make model using Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

for v in [var3,var4]:
    print("variable : %s" %v)
    X_train, X_test, y_train, y_test = divid_data(ad, v)
    print("y_test : ")
    print(y_test.value_counts())

    for d in [3,5,7,10]:
        print("When max_depth : %d :" %d)
        
        ## Train model Using Decision Tree
        tree = DecisionTreeClassifier(max_depth=d, random_state=1)
        tree.fit(X_train,y_train)

        ## predict is_attributed
        p = tree.predict_proba(X_test)[:,1]
        p = examine_outlier(p)

        ## Evaluate a model
        print("feature : %s" % tree.feature_importances_)
        print("coefficient of determination : %.5f" % tree.score(X_test,y_test))
        print("AUC : %.5f" % roc_auc_score(y_test, p))


## Make model using Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

for v in [var3,var4]:
    print("variable : %s" %v)
    X_train, X_test, y_train, y_test = divid_data(ad, v)
    print("y_test : ")
    print(y_test.value_counts())
    
    for d in [3,5,7,10]:
        for e in [30,50,70,100]:
            print("When max_depth : %d, n_estimators : %d ," %(d,e))
        
            ## Train model Using Decision Tree
            forest = RandomForestClassifier(max_depth=d, n_estimators=e, random_state=1)
            forest.fit(X_train,y_train)

            ## predict is_attributed
            p = forest.predict_proba(X_test)[:,1]
            p = examine_outlier(p)
        
            ## Evaluate a model
            print("feature : %s" % forest.feature_importances_)
            print("coefficient of determination : %.5f" % forest.score(X_test,y_test))
            print("AUC : %.5f" % roc_auc_score(y_test, p))


## Make model using Geadient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

for v in [var3,var4]:
    print("variable : %s" %v)
    X_train, X_test, y_train, y_test = divid_data(ad, v)
    print("y_test : ")
    print(y_test.value_counts())

    for x in [3,5,7]:
        for y in [0.01,0.1,1]:
            for z in [30,50,70]:
                print("when max_depth=%d, learning_rate=%.2f, n_estimators=%d : " %(x,y,z))
                
                ## Train model Using Gradient Boosting Classifier
                gbrt = GradientBoostingClassifier(max_depth=x, learning_rate=y, n_estimators=z, random_state=1)
                gbrt.fit(X_train,y_train)

                ## predict is_attributed
                p = gbrt.predict_proba(X_test)[:,1]
                p = examine_outlier(p)
            
                ## Evaluate a model              
                print("feature : %s" % gbrt.feature_importances_)
                print("accuracy : %.5f" % gbrt.score(X_test,y_test))
                print("AUC : %.5f" % roc_auc_score(y_test, p))
            

## Make model using Support Vector Machines
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

for v in [var3,var4]:
    print("variable : %s" %v)
    X_train, X_test, y_train, y_test = divid_data(ad, v)
    print("y_test : ")
    print(y_test.value_counts())

    for c in [0.1,1,10,100,1000]:
        for g in [0.1,1,10]:
            print("when C=%.2f, gamma=%.2f : " % (c,g))
            
            ## Train model Using Support vector Machines
            svm = SVC(C=c, gamma=g, probability=True, random_state=1)
            svm.fit(X_train,y_train)
        
            ## predict is_attributed
            p = svm.predict_proba(X_test)[:,1]
            p = examine_outlier(p)
        
            ## Evaluate a model
            print("accuracy : %.5f" % svm.score(X_test,y_test))
            print("AUC : %.5f" % roc_auc_score(y_test, p))
        
