
## 4. Target Variable Prediction - Linear Regression, Ridge, Logistic Regression
## Import library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


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
# ad = pd.read_csv('train_10m_modify1.csv')
# ad = pd.read_csv('train_20m_modify1.csv')
# ad = pd.read_csv('train_30m_modify1.csv')
ad = pd.read_csv('train_modify1.csv')
print(ad.columns)

ad_test = pd.read_csv('test_modify.csv')
print(ad_test.columns)

submission = pd.read_csv('sample_submission.csv')
print(submission.columns)


## Make a result DataFrame
i = 0
colnames = ['model','feat','param', 'auc','train','test']
result = pd.DataFrame(columns=colnames)


## Create features to use a model
feat1 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop','channel_attr_prop','hour_attr_prop','tot_attr_prop']
feat2 = ['ip_time_prop','ip_app_prop','ip_channel_prop','hour_app_prop','hour_channel_prop','tot_vv_prop']
feat3 = feat1 + feat2
feat4 = ['ip_attr_prop','app_attr_prop','channel_attr_prop','tot_attr_prop']
feat5 = feat4 + feat2
feat6 = ['app_attr_prop','channel_attr_prop','hour_app_prop','hour_channel_prop']


## Predict a target variable
feat = [feat1,feat2,feat3,feat4,feat5,feat6]
name = ['feat1','feat2','feat3','feat4','feat5','feat6']

for f,n in zip(feat,name):
    print("feat = %s" %n)
    
    ## Divid data    
    X_train, X_test, y_train, y_test = train_test_split(ad[f], ad['is_attributed'], random_state=1)

    print("X_train : " + str(X_train.shape))
    print("X_test : " + str(X_test.shape))
    print("y_train : " + str(y_train.shape))
    print("y_test : " + str(y_test.shape) + '\n')
    
    print("y_test : ")
    print(y_test.value_counts() + '\n')


    ## Make a model using Linear Regression
    ## Train a model
    lr = LinearRegression()
    lr.fit(X_train,y_train)

    ## predict is_attributed
    p = lr.predict(X_test)
    p, train = examine_outlier(p)

    ## Evaluate the model
    score = lr.score(X_test,y_test)
    auc = roc_auc_score(y_test, lr.predict(X_test))

    print("coefficient : %s" % lr.coef_)
    print("intercept : %s" % lr.intercept_)
    print("coefficient of determination : %.5f" % score)
    print("AUC : %.5f" % auc)

    ## Predict target variable
    is_attributed = lr.predict(ad_test[f])
    is_attributed, test = examine_outlier(is_attributed)

    ## Save result
    r = pd.DataFrame({'model':'lr',
                      'param':'null',
                      'feat':n,
                      'auc':auc,
                      'train':train,
                      'test':test}, columns=colnames, index=[i])
    result = pd.concat([result, r], ignore_index=True)
    i+=1    
    
    submission['is_attributed'] = is_attributed
    # submission.to_csv('10m_submission1_lr.csv', index=False)
    # submission.to_csv('20m_submission1_lr.csv', index=False)
    # submission.to_csv('30m_submission1_lr.csv', index=False)
    submission.to_csv('submission1_lr_' + n + '.csv', index=False)
    print("save complete..." + '\n')
    
    
    ## Make a model using Ridge    
    for a in [0.1,1,10]:
        print("When alpha=%.1f :" %a)
        
        ## Train a model
        ridge = Ridge(alpha=a)
        ridge.fit(X_train,y_train)
        
        ## predict is_attributed
        p = ridge.predict(X_test)
        p, train = examine_outlier(p)
        
        ## Evaluate the model
        score = ridge.score(X_test,y_test)
        auc = roc_auc_score(y_test, p)
        
        print("coefficient of determination : %.5f" % score)
        print("AUC : %.5f" % auc)
    
        ## Predict target variable
        is_attributed = ridge.predict(ad_test[f])
        is_attributed, test = examine_outlier(is_attributed)
    
        ## Save result
        r = pd.DataFrame({'model':'ridge',
                          'param':str(a),
                          'feat':n,
                          'auc':auc,
                          'train':train,
                          'test':test}, columns=colnames, index=[i])
        result = pd.concat([result, r], ignore_index=True)
        i+=1
        
        print("save complete..." + '\n')
    
    
    ## Make a model using Logistic Regression
    for c in [0.01,0.1,1,10]:
        print("When C=%.2f :" %c)
        
        ## Train a model
        logreg = LogisticRegression(C=c)
        logreg.fit(X_train,y_train)
    
        ## predict is_attributed
        p = logreg.predict_proba(X_test)[:,1]
        p, train = examine_outlier(p)
            
        ## Evaluate the model
        score = logreg.score(X_test,y_test)
        auc = roc_auc_score(y_test, p)
        
        print("coefficient of determination : %.5f" % score)
        print("AUC : %.5f" % auc)
    
        ## Predict target variable
        is_attributed = logreg.predict_proba(ad_test[f])[:,1]
        is_attributed, test = examine_outlier(is_attributed)
        
        ## Save result
        r = pd.DataFrame({'model':'logreg',
                          'param':str(c),
                          'feat':n,
                          'auc':auc,
                          'train':train,
                          'test':test}, columns=colnames, index=[i])
        result = pd.concat([result, r], ignore_index=True)
        i+=1    
            
        submission['is_attributed'] = is_attributed
        # submission.to_csv('10m_submission1_logreg_'+str(c)+'.csv', index=False)
        # submission.to_csv('20m_submission1_logreg_'+str(c)+'.csv', index=False)
        # submission.to_csv('30m_submission1_logreg_'+str(c)+'.csv', index=False)
        submission.to_csv('submission1_logreg_' + str(c) + '_' + n + '.csv', index=False)
        print("save complete..." + '\n')
    

## Save resultset
result.to_csv('result.csv', index=False)