
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


## Makes a model using Logistic Regression
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
    
    coef = log.coef_
    result_data = [c, train_auc, valid_auc] + list(coef)
    
    print("AUC of train data : %.5f" % train_auc)
    print("AUC of valid data : %.5f" % valid_auc)
    print(coef)
    
    ## Predict target variable
    test = pd.read_csv("data/test_add_features.csv", usecols=feat + ['is_attributed'])
    pred = log.predict_proba(test[feat])[:,1]      
    
    return pred, result_data


## Makes a model using Decision Tree
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
    
    importances = tree.feature_importances_
    result_data = [max_depth, train_auc, valid_auc] + list(importances)
    
    print("AUC of train data : %.5f" % train_auc)
    print("AUC of valid data : %.5f" % valid_auc)
    print(importances)
    
    ## Predict target variable
    test = pd.read_csv("data/test_add_features.csv", usecols=feat + ['is_attributed'])
    pred = tree.predict_proba(test[feat])[:,1]      
    
    return pred, result_data


## Makes a model using Random Forest
def forest(feat, max_depth, n_estimators):
    print("When max_depth=%d, n_estimators=%d :" % (max_depth, n_estimators))    
    
    ## Divid Dataset
    X_train, X_valid, y_train, y_valid = divid_data(feat)
    
    ## Train the model
    fst = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    fst.fit(X_train, y_train)
    
    ## Evaluate the model
    p = fst.predict_proba(X_train)[:,1]
    train_auc = roc_auc_score(y_train, p)
    
    p = fst.predict_proba(X_valid)[:,1]
    valid_auc = roc_auc_score(y_valid, p)
    
    importances = fst.feature_importances_
    result_data = [max_depth, n_estimators, train_auc, valid_auc] + list(importances)
    
    print("AUC of train data : %.5f" % train_auc)
    print("AUC of valid data : %.5f" % valid_auc)
    print(importances)
    
    ## Predict target variable
    test = pd.read_csv("data/test_add_features.csv", usecols=feat + ['is_attributed'])
    pred = fst.predict_proba(test[feat])[:,1]     
    
    return pred, result_data


## Makes a model using Gradient Boosting
def boost(feat, max_depth, n_estimators, learning_rate):
    print("when max_depth=%d, n_estimators=%d, learning_rate=%.2f : " %(max_depth, n_estimators, learning_rate))
    
    ## Divid Dataset
    X_train, X_valid, y_train, y_valid = divid_data(feat)
    
    ## Train the model
    bst = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
    bst.fit(X_train,y_train)
    
    ## Evaluate the model
    p = bst.predict_proba(X_train)[:,1]
    train_auc = roc_auc_score(y_train, p)
    
    p = bst.predict_proba(X_valid)[:,1]
    valid_auc = roc_auc_score(y_valid, p)
    
    importances = bst.feature_importances_
    result_data = [max_depth, n_estimators, learning_rate, train_auc, valid_auc] + list(importances)
    
    print("AUC of train data : %.5f" % train_auc)
    print("AUC of valid data : %.5f" % valid_auc)
    print(importances)
    
    ## Predict target variable
    test = pd.read_csv("data/test_add_features.csv", usecols=feat + ['is_attributed'])
    pred = bst.predict_proba(test[feat])[:,1]      
    
    return pred, result_data


## Makes a model using LightGBM
def lgbm(feat, max_depth, learning_rate):
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
        'min_split_gain': 0, 
        'reg_alpha': 0,
        'reg_lambda': 0, 
        'nthread': 4,
        'verbose': 0,
        'metric':'auc',     
        
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_child_samples': 100,  
        'max_bin': 100, 
        'subsample': 0.7, 
        'subsample_freq': 1, 
        'colsample_bytree': 0.9, 
        'min_child_weight': 0, 
        'scale_pos_weight':99
    }
    
    bst = lgb.train(params, train_set, 
                    valid_sets=[train_set,valid_set],
                    valid_names=['train','valid'],
                    num_boost_round=1000,
                    early_stopping_rounds=50, 
                    verbose_eval=100,
                    feval=None)
    
    ## Evaluate the model    
    score = bst.best_score()
    
    import_split = bst.feature_importance()
    import_split = import_split / import_split.sum()
    
    import_gain = bst.feature_importance('gain')
    import_gain = import_gain / import_gain.sum()
    
    result_data = [max_depth, learning_rate] + list(score) + list(import_gain)
    
    print(score)
    print(import_split)
    print(import_gain)
    
    ## Predict the target
    test = pd.read_csv("data/test_add_features.csv", usecols=feat + ['is_attributed'])
    pred = bst.predict(test[feat], num_iteration=bst.best_iteration)
    
    return pred, result_data


## Save predicted data
def save(predict, filename):
    submission = pd.read_csv('submission/sample_submission.csv')
    submission['is_attributed'] = predict
    submission.to_csv('submission/' + filename + '.csv', index=False)
  

## ------------------------------------------- rate -------------------------------------------
## 1
feat = ['rate_ip', 'rate_app', 'rate_os', 'rate_device', 'rate_channel', 'rate_hour']

# Logistic Regression
col = ['C','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  

for c in [0.01, 0.1, 1, 10, 100]:
    pred, result_data = logistic(feat, c)
    save(pred, 'log_' + str(c))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('log_result.csv', index=False)

# Decision Tree
col = ['max_depth','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  
    
for d in range(3,8):
    pred, result_data = tree(feat, d)
    save(pred, 'tree_' + str(d))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('tree_result.csv', index=False)

# Random Forest
col = ['max_depth','n_estimators','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)   
    
for d in range(3,8):
    for n in [50, 70, 100]:
        pred, result_data = forest(feat, d, n)
        save(pred, 'fst_'+'_'.join([str(d),str(n)]))
        result_data = pd.DataFrame([result_data], columns=col)
        result = pd.concat([result, result_data])
        result.to_csv('fst_result.csv', index=False)
         

## 2
feat = ['rate_app', 'rate_os', 'rate_device', 'rate_channel', 'rate_hour']

# Logistic Regression
col = ['C','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  

for c in [0.01, 0.1, 1, 10, 100]:
    pred, result_data = logistic(feat, c)
    save(pred, 'log_' + str(c))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('log_result.csv', index=False)

# Decision Tree
col = ['max_depth','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  
    
for d in range(3,8):
    pred, result_data = tree(feat, d)
    save(pred, 'tree_' + str(d))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('tree_result.csv', index=False)

# Random Forest
col = ['max_depth','n_estimators','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)   
    
for d in range(3,8):
    for n in [50, 70, 100]:
        pred, result_data = forest(feat, d, n)
        save(pred, 'fst_'+'_'.join([str(d),str(n)]))
        result_data = pd.DataFrame([result_data], columns=col)
        result = pd.concat([result, result_data])
        result.to_csv('fst_result.csv', index=False)
    
## 3
feat = ['rate_app','rate_os', 'rate_channel']

# Logistic Regression
col = ['C','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  

for c in [0.01, 0.1, 1, 10, 100]:
    pred, result_data = logistic(feat, c)
    save(pred, 'log_' + str(c))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('log_result.csv', index=False)

# Decision Tree
col = ['max_depth','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  
    
for d in range(3,8):
    pred, result_data = tree(feat, d)
    save(pred, 'tree_' + str(d))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('tree_result.csv', index=False)

# Random Forest
col = ['max_depth','n_estimators','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)   
    
for d in range(3,8):
    for n in [50, 70, 100]:
        pred, result_data = forest(feat, d, n)
        save(pred, 'fst_'+'_'.join([str(d),str(n)]))
        result_data = pd.DataFrame([result_data], columns=col)
        result = pd.concat([result, result_data])
        result.to_csv('fst_result.csv', index=False)
        
# Gradient Boosting
col = ['max_depth','n_estimators','learning_rate','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)

for d in range(5,8):
    for n in [50, 70]:
        for l in [0.001, 0.01, 0.1]:
            pred, result_data = boost(feat, d, n, l)
            save(pred, 'bst_'+'_'.join([str(d),str(n),str(l)]))
            result_data = pd.DataFrame([result_data], columns=col)
            result = pd.concat([result, result_data])
            result.to_csv('bst_result.csv', index=False)

## LightGBM   
for d in range(3,8):
    pred, result_data = lgbm(feat, 3, 0.05)
    save(pred, 'lgb_'+str(d))
    

## ------------------------------------------- gap -------------------------------------------
## 1
feat = ['gap_ip', 'gap_app', 'gap_device', 'gap_os', 'gap_channel']

# Logistic Regression
col = ['C','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  

for c in [0.01, 0.1, 1, 10, 100]:
    pred, result_data = logistic(feat, c)
    save(pred, 'log_' + str(c))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('log_result.csv', index=False)

# Decision Tree
col = ['max_depth','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  
    
for d in range(3,8):
    pred, result_data = tree(feat, d)
    save(pred, 'tree_' + str(d))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('tree_result.csv', index=False) 


## 2
feat = ['gap_ip', 'gap_app', 'gap_channel']

# Logistic Regression
col = ['C','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  

for c in [0.01, 0.1, 1, 10, 100]:
    pred, result_data = logistic(feat, c)
    save(pred, 'log_' + str(c))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('log_result.csv', index=False)

# Decision Tree
col = ['max_depth','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  
    
for d in range(3,8):
    pred, result_data = tree(feat, d)
    save(pred, 'tree_' + str(d))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('tree_result.csv', index=False)


## ------------------------------------------- black -------------------------------------------
feat = ['black_ip', 'black_app', 'black_device', 'black_os', 'black_channel']

# Logistic Regression
col = ['C','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  

for c in [0.01, 0.1, 1, 10, 100]:
    pred, result_data = logistic(feat, c)
    save(pred, 'log_' + str(c))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('log_result.csv', index=False)

# Decision Tree
col = ['max_depth','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  
    
for d in range(3,8):
    pred, result_data = tree(feat, d)
    save(pred, 'tree_' + str(d))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('tree_result.csv', index=False)



## ------------------------------------------- all -------------------------------------------
## 1
feat = ['rate_ip', 'rate_app', 'rate_device', 'rate_os', 'rate_channel', 'rate_hour',
        'gap_ip', 'gap_app', 'gap_device', 'gap_os', 'gap_channel', 'gap_hour',
        'black_ip', 'black_app', 'black_device', 'black_os', 'black_channel', 'black_hour',
        'click_gap']

# Decision Tree
col = ['max_depth','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  
    
for d in range(3,8):
    pred, result_data = tree(feat, d)
    save(pred, 'tree_' + str(d))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('tree_result.csv', index=False)

    

## 2
feat = ['rate_app', 'rate_device', 'rate_os', 'rate_channel', 'rate_hour',
        'gap_ip', 'gap_app', 'gap_device', 'gap_os', 'gap_channel', 'gap_hour',
        'black_ip', 'black_app', 'black_device', 'black_os', 'black_channel', 'black_hour',
        'click_gap']

col = ['max_depth','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  
    
for d in range(3,8):
    pred, result_data = tree(feat, d)
    save(pred, 'tree_' + str(d))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('tree_result.csv', index=False)
    

## 3
feat = ['rate_app', 'rate_device', 'rate_os', 'rate_channel', 'rate_hour',
        'gap_app', 'gap_device', 'gap_os', 'gap_channel', 'gap_hour',
        'black_ip', 'black_app', 'black_device', 'black_os', 'black_channel', 'black_hour',
        'click_gap']

# Decision Tree
col = ['max_depth','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  
    
for d in range(3,8):
    pred, result_data = tree(feat, d)
    save(pred, 'tree_' + str(d))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('tree_result.csv', index=False)

# Random Forest
col = ['max_depth','n_estimators','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)   
    
for d in range(3,8):
    for n in [50, 70, 100]:
        pred, result_data = forest(feat, d, n)
        save(pred, 'fst_'+'_'.join([str(d),str(n)]))
        result_data = pd.DataFrame([result_data], columns=col)
        result = pd.concat([result, result_data])
        result.to_csv('fst_result.csv', index=False)
        
# Gradient Boosting
col = ['max_depth','n_estimators','learning_rate','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)

for d in range(5,8):
    for n in [50, 70]:
        for l in [0.001, 0.01, 0.1]:
            pred, result_data = boost(feat, d, n, l)
            save(pred, 'bst_'+'_'.join([str(d),str(n),str(l)]))
            result_data = pd.DataFrame([result_data], columns=col)
            result = pd.concat([result, result_data])
            result.to_csv('bst_result.csv', index=False)
            
for d in range(3,8):
    pred, bst = lgbm(feat, d, 0.01)
    a = bst.feature_importance('gain')
    print(a / a.sum())
    save(pred, 'lgb_'+str(d))
    

## 4
feat = ['rate_app', 'rate_os', 'rate_device', 'rate_channel',
        'gap_app', 'gap_device', 'gap_os', 'gap_channel', 
        'black_ip', 'black_app', 'black_os', 'black_channel', 'click_gap']

# Decision Tree
col = ['max_depth','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  
    
for d in range(3,8):
    pred, result_data = tree(feat, d)
    save(pred, 'tree_' + str(d))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('tree_result.csv', index=False)

# Random Forest
col = ['max_depth','n_estimators','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)   
    
for d in range(3,8):
    for n in [50, 70, 100]:
        pred, result_data = forest(feat, d, n)
        save(pred, 'fst_'+'_'.join([str(d),str(n)]))
        result_data = pd.DataFrame([result_data], columns=col)
        result = pd.concat([result, result_data])
        result.to_csv('fst_result.csv', index=False)
        
# Gradient Boosting
col = ['max_depth','n_estimators','learning_rate','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)

for d in range(5,8):
    for n in [50, 70]:
        for l in [0.001, 0.01, 0.1]:
            pred, result_data = boost(feat, d, n, l)
            save(pred, 'bst_'+'_'.join([str(d),str(n),str(l)]))
            result_data = pd.DataFrame([result_data], columns=col)
            result = pd.concat([result, result_data])
            result.to_csv('bst_result.csv', index=False)


## 5
feat = ['rate_app', 'rate_channel', 'rate_hour', 
        'gap_app', 'gap_device', 'gap_os', 'gap_channel',
        'black_ip', 'click_gap']

# Decision Tree
col = ['max_depth','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  
    
for d in range(3,8):
    pred, result_data = tree(feat, d)
    save(pred, 'tree_' + str(d))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('tree_result.csv', index=False)

# Random Forest
col = ['max_depth','n_estimators','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)   
    
for d in range(3,8):
    for n in [50, 70, 100]:
        pred, result_data = forest(feat, d, n)
        save(pred, 'fst_'+'_'.join([str(d),str(n)]))
        result_data = pd.DataFrame([result_data], columns=col)
        result = pd.concat([result, result_data])
        result.to_csv('fst_result.csv', index=False)
        
# Gradient Boosting
col = ['max_depth','n_estimators','learning_rate','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)

for d in range(5,8):
    for n in [50, 70]:
        for l in [0.001, 0.01, 0.1]:
            pred, result_data = boost(feat, d, n, l)
            save(pred, 'bst_'+'_'.join([str(d),str(n),str(l)]))
            result_data = pd.DataFrame([result_data], columns=col)
            result = pd.concat([result, result_data])
            result.to_csv('bst_result.csv', index=False)

for d in range(3,8):
    pred, bst = lgbm(feat, d, 0.05)
    a = bst.feature_importance('gain')
    print(a / a.sum())
    save(pred, 'lgb_'+str(d))


## 6
feat = ['rate_app', 'rate_os', 'rate_channel', 'gap_app', 'gap_channel', 
        'black_ip', 'black_device', 'click_gap']

# Decision Tree
col = ['max_depth','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)  
    
for d in range(3,8):
    pred, result_data = tree(feat, d)
    save(pred, 'tree_' + str(d))
    result_data = pd.DataFrame([result_data], columns=col)
    result = pd.concat([result, result_data])
    result.to_csv('tree_result.csv', index=False)

# Random Forest
col = ['max_depth','n_estimators','Train AUC','Valid AUC'] + feat
result = pd.DataFrame(columns=col)   
    
for d in range(3,8):
    for n in [50, 70, 100]:
        pred, result_data = forest(feat, d, n)
        save(pred, 'fst_'+'_'.join([str(d),str(n)]))
        result_data = pd.DataFrame([result_data], columns=col)
        result = pd.concat([result, result_data])
        result.to_csv('fst_result.csv', index=False)
        
for d in range(3,8):
    pred, bst = lgbm(feat, d, 0.05)
    a = bst.feature_importance('gain')
    print(a / a.sum())
    save(pred, 'lgb_'+str(d))


## ------------------------------------------- stacking -------------------------------------------
import pandas as pd
import numpy as np    

result1 = pd.read_csv('submission/lgb_40m_3.csv', usecols=['is_attributed'])
result2 = pd.read_csv('submission/lgb_40m_4.csv', usecols=['is_attributed']) 
result3 = pd.read_csv('submission/lgb_40m_5.csv', usecols=['is_attributed']) 
result4 = pd.read_csv('submission/lgb_40m_6.csv', usecols=['is_attributed']) 
result5 = pd.read_csv('submission/lgb_40m_7.csv', usecols=['is_attributed'])

result = pd.concat([result1, result2, result3, result4, result5], axis=1)
result.columns = [3, 4, 5, 6, 7]
print(result.head())
print(result.tail())

result['mean'] = result.mean(axis=1)
print(result.head())
print(result.tail())
save(result['mean'], 'stk_mean')
    
for i in np.linspace(0, 1, 11):
    feat = 'min_max_' +str(round(i,1))
    result[feat] = result[[3, 4, 5, 6, 7]].min(axis=1)
    result.loc[result['mean'] > i, feat] = result.loc[result['mean'] > i, [3, 4, 5, 6, 7]].max(axis=1)
    save(result[feat], 'stk_' + feat)
    
print(result.head())
print(result.tail())
result.to_csv('stacking.csv', index=False)


