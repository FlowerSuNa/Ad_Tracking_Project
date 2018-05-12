
## 3-3. Target Variable Prediction - K-Nearest Neighbors, Support Vector Machines
## Import library
import pandas as pd
from sklearn.model_selection import train_test_split


## Create functions
def check_data(is_attributed):
    count = [0,0,0,0,0,0,0,0,0,0,0,0]
    
    for i in range(len(is_attributed)):
        if is_attributed[i] > 1:
            count[11] += 1
        elif is_attributed[i] > 0.9:
            count[10] += 1
        elif is_attributed[i] > 0.8:
            count[9] += 1
        elif is_attributed[i] > 0.7:
            count[8] += 1
        elif is_attributed[i] > 0.6:
            count[7] += 1
        elif is_attributed[i] > 0.5:
            count[6] += 1
        elif is_attributed[i] > 0.4:
            count[5] += 1
        elif is_attributed[i] > 0.3:
            count[4] += 1
        elif is_attributed[i] > 0.2:
            count[3] += 1
        elif is_attributed[i] > 0.1:
            count[2] += 1
        elif is_attributed[i] >= 0:
            count[1] += 1
        else:
            count[0] += 1
         
    count = ' '.join(str(x) for x in count)
    print(count)
    
    return count

    
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


## Import data
ad = pd.read_csv('train_10m_modify1.csv')
print(ad.columns)

ad_test = pd.read_csv('test_modify1.csv')
print(ad_test.columns)

submission = pd.read_csv('sample_submission.csv')
print(submission.columns)


## Create features to use a model
feat = ['app_attr_prop','channel_attr_prop','hour_app_prop','hour_channel_prop']


## Divid data
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

    ## Train the model
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train,y_train)

    ## predict is_attributed
    p = knn.predict(X_test)
    p, train = examine_outlier(p)

    ## Evaluate the model  
    print("coefficient of determination : %.5f" % knn.score(X_test,y_test))
    print("AUC : %.5f" % roc_auc_score(y_test,p))

    ## Predict target variable
    is_attributed = knn.predict(ad_test[feat])
    is_attributed, test = examine_outlier(is_attributed)


## Make a model using Support Vector Machines
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

for c in [0.1,1,10,100,1000]:
    for g in [0.1,1,10]:
        print("when C=%.1f , gamma=%.1f :" % (c,g))
    
        ## Train the model
        svm = SVC(C=c, gamma=g, probability=True, random_state=1)
        svm.fit(X_train,y_train)
    
        ## predict is_attributed
        p = svm.predict_proba(X_test)[:,1]
        p, train = examine_outlier(p)
    
        ## Evaluate the model
        score = svm.score(X_test,y_test)
        auc = roc_auc_score(y_test, p)
        
        print("coefficient of determination : %.5f" % score)
        print("AUC : %.5f" % auc)
    
        ## Predict target variable
        is_attributed = svm.predict_proba(ad_test[feat])[:,1]
        is_attributed, test = examine_outlier(is_attributed)