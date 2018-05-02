
## 4. Target Variable Prediction - xgBoost, Lightgbm
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
colnames = ['name','param', 'auc','data']
result = pd.DataFrame(columns=colnames)


## Check correlation
feat1 = ['app_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop',
         'channel_attr_prop','tot_attr_prop']

feat2 = ['ip_time_prop','ip_app_prop','ip_channel_prop','time_app_prop',
         'time_channel_prop','tot_vv_prop']

feat3 = feat1 + feat2

feat4 = ['app_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop',
         'channel_attr_prop','tot_attr_prop']

feat5 = ['time_app_prop','time_channel_prop','tot_vv_prop']

feat6 = feat4 + feat5

print(ad[feat3 + ['is_attributed']].corr(method='pearson'))


## Select features
# feat = feat1
# feat = feat2
# feat = feat3
feat = feat4
# feat = feat5
# feat = feat6


## Divid data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(ad[feat], ad['is_attributed'], random_state=1)

print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))


## Make a model using xgBoost
import xgboost as xgb

train = xgb.DMatrix(X_train,y_train)
test = xgb.DMatrix(X_test,y_test)

params = {'eta':0.3, 'tree_method':'hist', 'grow_policy':'lossguide',
          'max_leaves':1400, 'max_depth':0, 'subsample':0.9, 
          'colsample_bytree':0.7, 'colsample_bylevel':0.7, 'min_child_weight':0,
          'alpha':4, 'objective':'binary:logistic', 'scale_pos_weight':9,
          'eval_metric':'auc', 'nthread':8, 'random_state':1, 'silent':True}
          
model = xgb.train(params, train, 200)


## Make a model using lightgbm
import lightgbm as lgb

train = lgb.Dataset(X_train.values, label=y_train.values, feature_name=feat)
valid = lgb.Dataset(X_test.values, label=y_test.values, feature_name=feat)

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

bst = lgb.train(params, train, 
                valid_sets=[train,valid],
                valid_names=['train','valid'],
                num_boost_round=350,
                early_stopping_rounds=30, 
                verbose_eval=True,
                
                feval=None)

## Save result
lgb.plot_importance(bst)

is_attributed = bst.predict(ad_test[feat], num_iteration=bst.best_iteration)
is_attributed, test = examine_outlier(is_attributed)

submission['is_attributed'] = is_attributed
submission.to_csv('10m_submission_gbm_feat4.csv', index=False)
