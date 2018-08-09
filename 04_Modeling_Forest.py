
## 4. Modeling
## Import library
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


## Load data
submission = pd.read_csv('submission/sample_submission.csv')


## Divid data
def divid_data(feat):
    train = pd.read_csv("data/train_add_features_20m.csv", usecols=feat + ['is_attributed'])
    
    X_train, X_valid, y_train, y_valid = train_test_split(train[feat], train['is_attributed'], 
                                                          random_state=0, test_size=0.2)
    
    print("X_train : ", X_train.shape)
    print("X_valid : ", X_valid.shape)
    print("y_train : ", y_train.shape)
    print("y_valid : ", y_valid.shape)
    
    print("download frequency of y_train : ")
    print(y_train.value_counts())
    
    print("download frequency of y_valid : ")
    print(y_valid.value_counts())
    
    return X_train, X_valid, y_train, y_valid


## Make a model using Decision Tree
def forest(feat, max_depth, n_estimators, max_features):
    print("When max_depth=%d, n_estimators=%d, max_features=%d :" % (max_depth, n_estimators, max_features))    
    
    ## Divid Dataset
    X_train, X_valid, y_train, y_valid = divid_data(feat)
    
    ## Train the model
    forest = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=max_features)
    forest.fit(X_train, y_train)
    
    ## Evaluate the model
    p = forest.predict_proba(X_train)[:,1]
    train_auc = roc_auc_score(y_train, p)
    
    p = forest.predict_proba(X_valid)[:,1]
    valid_auc = roc_auc_score(y_valid, p)
    
    print("AUC of train data : %.5f" % train_auc)
    print("AUC od valid data : %.5f" % valid_auc)
    
    ## Predict target variable
    test = pd.read_csv("data/test_add_features.csv", usecols=feat + ['is_attributed'])
    pred = forest.predict_proba(test[feat])[:,1]      
    
    return pred, forest

feat = ['black_ip', 'black_app', 'black_os', 'black_channel']
for d in range(1,6):
    pred, forest_ = forest(feat, d)
    print(forest_.feature_importances_)

submission['is_attributed'] = pred
submission.to_csv('submission/submission_forest.csv', index=False)

