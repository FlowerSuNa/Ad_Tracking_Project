
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
    
    return pred

feat = ['black_ip', 'black_app', 'black_device', 'black_os']
pred = logistic(train, test, feat, result)
submission['is_attributed'] = pred
submission.to_csv('submission_logistic_' + str(c) + '_black.csv', index=False)


feat = ['gap_ip', 'black_ip', 'gap_app', 'black_app', 
        'gap_device', 'black_device', 'gap_os', 'black_os']
pred = logistic(train, test, feat, result)
submission['is_attributed'] = pred
submission.to_csv('submission_logistic_' + str(c) + '_gap_black.csv', index=False)
