
## 3. Modeling
## Import library
import pandas as pd
import gc

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


## Load data
data = pd.read_csv("merge_gap_black.csv", parse_dates=['click_time'])
submission = pd.read_csv('sample_submission.csv')
gc.collect()


## Divid data
def divid_data(df, feat):
    X_train, X_valid, y_train, y_valid = train_test_split(df[feat], df['is_attributed'], 
                                                          random_state=0, test_size=0.2)
    
    print("X_train : " + str(X_train.shape))
    print("X_valid : " + str(X_valid.shape))
    print("y_train : " + str(y_train.shape))
    print("y_valid : " + str(y_valid.shape) + '\n')
    
    print("download frequency of y_train : ")
    print(y_train.value_counts())
    
    print("download frequency of y_valid : ")
    print(y_valid.value_counts())
    
    return X_train, X_valid, y_train, y_valid

train = data.loc[data['click_id'].isnull()]
test = data.loc[data['click_id'].notnull()]

del data
gc.collect()


## Make a result DataFrame
colnames = ['model','feat','param', 'train auc','valid auc']
result = pd.DataFrame(columns=colnames)


## Make a model using Logistic Regression
def logistic(train_df, test_df, feat, c=1, result):
    print("When C=%.2f :" %c)    
    
    ## Divid Dataset
    X_train, X_valid, y_train, y_valid = divid_data(train_df, feat)
    
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
    
    ## Save result
    f = ', '.join(feat)
    r = pd.DataFrame({'model':'logistic',
                      'param':str(c),
                      'feat':f,
                      'train auc':train_auc,
                      'valid auc':valid_auc}, columns=colnames, index=0)
    result = pd.concat([result, r], ignore_index=True)
        
    ## Predict target variable
    pred = log.predict_proba(test_df[feat])[:,1]      
    
    return pred, result

feat = ['black_ip', 'black_app', 'black_device', 'black_os']
pred, result = logistic(train, test, feat, result)
submission['is_attributed'] = pred
submission.to_csv('submission_logistic_' + str(c) + '_black.csv', index=False)


feat = ['gap_ip', 'black_ip', 'gap_app', 'black_app', 
        'gap_device', 'black_device', 'gap_os', 'black_os']
pred, result = logistic(train, test, feat, result)
submission['is_attributed'] = pred
submission.to_csv('submission_logistic_' + str(c) + '_gap_black.csv', index=False)


##
def lgbm(train_df, test_df, feat, categorical):
    ## Divid Dataset
    X_train, X_valid, y_train, y_valid = divid_data(train_df, feat)
    
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
    is_attributed = bst.predict(test_df[feat], num_iteration=bst.best_iteration)
    
    return is_attributed

feat = ['black_ip', 'black_app', 'black_device', 'black_os']
categorical = []
pred = logistic(train, test, feat, categorical)
submission['is_attributed'] = pred
submission.to_csv('submission_lightgbm_' + str(c) + '_black.csv', index=False)