
## 3. Target Variable Prediction - Lightgbm
## Import library
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc


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


def lgbm(train_data, test_data, feat, target):
    ## Divid data    
    X_train, X_test, y_train, y_test = train_test_split(train_data[feat], train_data[target], random_state=1)    
    print("X_train : " + str(X_train.shape))
    print("X_test : " + str(X_test.shape))
    print("y_train : " + str(y_train.shape))
    print("y_test : " + str(y_test.shape))
    
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
    
    ## Check result
    is_attributed = bst.predict(test_data[feat], num_iteration=bst.best_iteration)
    is_attributed, test_result = examine_outlier(is_attributed)
    
    return is_attributed, test_result


## Import submission dataset
submission = pd.read_csv('sample_submission.csv')
print(submission.shape)
print(submission.columns)


## Make a result DataFrame
i = 0
colnames = ['sample','features', 'test']
result = pd.DataFrame(columns=colnames)


## Make a model using lightgbm
## Compose features
feat1 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop','channel_attr_prop','hour_attr_prop','tot_attr_prop']
feat2 = ['ip_hour_prop','ip_app_prop','ip_channel_prop','hour_app_prop','hour_channel_prop','tot_vv_prop']
feat3 = ['ip_attr_tot_prop','app_attr_tot_prop','device_attr_tot_prop','os_attr_tot_prop','channel_attr_tot_prop','hour_attr_tot_prop','tot_attr_tot_prop']
feat4 = ['app_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop','channel_attr_prop']
feat5 = feat1 + feat2
feat6 = feat1 + feat2 + feat3
feat7 = feat4 + ['hour_app_prop','hour_channel_prop']

target = 'is_attributed'
feat = [feat1, feat2, feat3, feat4, feat5, feat6, feat7]
ft = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5', 'feat6', 'feat7']
sample = ['10m','20m','30m','40m','50m']

## Import test data
ad_test = pd.read_csv('test_modify.csv')
print(ad_test.columns)

for s in sample:
    print('sample : %s' % s)
    
    ## Import train data
    ad_train = pd.read_csv('train_modify_'+ s + '.csv')
    print(ad_train.columns)    
    
    for f,t in zip(feat,ft):
        print('features : %s' % f)
        
        ## Train a model
        is_attributed, test = lgbm(ad_train, ad_test, f, target)
        gc.collect()
        
        ## Append result data
        r = pd.DataFrame({'sample':s,'features':t, 'test':test}, columns=colnames, index=[i])
        result = pd.concat([result, r], ignore_index=True)
        i+=1
        
        ## Save submission dataset
        submission['is_attributed'] = is_attributed
        submission.to_csv(s + '_submission_gbm_' + t + '.csv', index=False)
    
    ## Remove dataset
    del ad_train
    del ad_test


## Save result dataset
result.to_csv('result.csv', index=False)