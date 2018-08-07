
## 3. Modeling
## Import library
import pandas as pd
import gc

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import lightgbm as lgb


## Load data
submission = pd.read_csv('submission/sample_submission.csv')
gc.collect()


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


##
def lgbm(feat, categorical):
    ## Divid Dataset
    X_train, X_valid, y_train, y_valid = divid_data(feat)
    
    train_set = lgb.Dataset(X_train.values, 
                            label=y_train.values, 
                            feature_name=feat, 
                            categorical_feature=categorical)
    valid_set = lgb.Dataset(X_valid.values, 
                            label=y_valid.values, 
                            feature_name=feat, 
                            categorical_feature=categorical)
    
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
feat = ['black_ip', 'gap_app', 'black_app', 'gap_device',
        'gap_os', 'black_os', 'gap_channel', 'black_channel']
categorical = ['black_ip', 'black_app', 'black_os', 'black_channel']
pred, bst = lgbm(feat, categorical)

bst.feature_importance()
submission['is_attributed'] = pred
submission.to_csv('submission_lightgbm.csv', index=False)
