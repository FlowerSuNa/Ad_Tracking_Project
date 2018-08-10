
## 4. Modeling
## Import library
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import lightgbm as lgb


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


## Make a model using Logistic Regression
def logistic(feat, c=1):
    print("When C=%.2f :" %c)    
    
    ## Divid Dataset
    X_train, X_valid, y_train, y_valid = divid_data(feat)
    
    ## Train the model
    log = LogisticRegression(C=c)
    log.fit(X_train, y_train)
    
    ## Evaluate the model
    p = log.predict_proba(X_train)[:,1]
    train_auc = roc_auc_score(y_train, p)
    
    p = log.predict_proba(X_valid)[:,1]
    valid_auc = roc_auc_score(y_valid, p)
    
    print("AUC of train data : %.5f" % train_auc)
    print("AUC od valid data : %.5f" % valid_auc)
    
    ## Predict target variable
    test = pd.read_csv("data/test_add_features.csv", usecols=feat + ['is_attributed'])
    pred = log.predict_proba(test[feat])[:,1]      
    
    return pred, log


## Make a model using Decision Tree
def tree(feat, max_depth):
    print("When max_depth=%d :" % max_depth)    
    
    ## Divid Dataset
    X_train, X_valid, y_train, y_valid = divid_data(feat)
    
    ## Train the model
    tree = DecisionTreeClassifier(max_depth=max_depth)
    tree.fit(X_train, y_train)
    
    ## Evaluate the model
    p = tree.predict_proba(X_train)[:,1]
    train_auc = roc_auc_score(y_train, p)
    
    p = tree.predict_proba(X_valid)[:,1]
    valid_auc = roc_auc_score(y_valid, p)
    
    print("AUC of train data : %.5f" % train_auc)
    print("AUC od valid data : %.5f" % valid_auc)
    
    ## Predict target variable
    test = pd.read_csv("data/test_add_features.csv", usecols=feat + ['is_attributed'])
    pred = tree.predict_proba(test[feat])[:,1]      
    
    return pred, tree


## Make a model using Random Forest
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


def lgbm(feat):
    ## Divid Dataset
    X_train, X_valid, y_train, y_valid = divid_data(feat)
    
    train_set = lgb.Dataset(X_train.values, 
                            label=y_train.values, 
                            feature_name=feat)
    valid_set = lgb.Dataset(X_valid.values, 
                            label=y_valid.values, 
                            feature_name=feat)
    
    ## train a model
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'min_split_gain': 0,    # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,         # L1 regularization term on weights
        'reg_lambda': 0,        # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric':'auc',     
     
        'learning_rate': 0.15,
        'num_leaves': 7,        # 2^max_depth - 1
        'max_depth': 3,         # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,         # Number of bucketed bin for feature values
        'subsample': 0.7,       # Subsample ratio of the training instance.
        'subsample_freq': 1,    # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':99
    }
    
    bst = lgb.train(params, train_set, 
                    valid_sets=[train_set,valid_set],
                    valid_names=['train','valid'],
                    num_boost_round=1000,
                    early_stopping_rounds=50, 
                    verbose_eval=100,
                    feval=None)
    
    ## Predict the target
    test = pd.read_csv("data/test_add_features.csv", usecols=feat + ['is_attributed'])
    pred = bst.predict(test[feat], num_iteration=bst.best_iteration)
    
    return pred, bst


##
def save(predict, filename):
    submission = pd.read_csv('submission/sample_submission.csv')
    submission['is_attributed'] = pred
    submission.to_csv('submission/' + filename + '.csv', index=False)


## 
feat = ['black_ip', 'black_app', 'black_os', 'black_channel']
for c in [0.01, 0.1, 1, 10, 100]:
    pred, log_ = logistic(feat, c)
    print(log_.coef_)
    save(pred, 'log_' + str(c))
    
for d in range(1,6):
    pred, tree_ = tree(feat, d)
    print(tree_.feature_importances_)
    save(pred, 'tree_' + str(d))
    
pred, bst = lgbm(feat)
a = bst.feature_importance('gain')
print(a / a.sum())
save(pred, 'lgb')



##
feat = ['black_ip', 'gap_app', 'black_app', 'gap_os', 'black_os', 
        'gap_channel', 'black_channel']
for c in [0.1, 1, 10]:
    pred, log_ = logistic(feat, c)
    print(log_.coef_)
    save(pred, 'log_' + str(c))

for d in range(3,8):
    pred, tree_ = tree(feat, d)
    print(tree_.feature_importances_)
    save(pred, 'tree_' + str(d))

pred, bst = lgbm(feat)
a = bst.feature_importance('gain')
print(a / a.sum())
save(pred, 'lgb')


##
feat = ['black_ip', 'gap_app', 'black_app', 'gap_os', 'black_os', 
        'gap_channel', 'black_channel', 'black_hour', 'click_gap']
pred, log_ = logistic(feat, c)
print(log_.coef_)
save(pred, 'log_' + str(c))


