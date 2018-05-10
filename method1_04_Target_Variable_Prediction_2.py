
## 4. Target Variable Prediction - Decision Tree, Random Forest, Gradient Boosting
## Import library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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

ad_test = pd.read_csv('test_modify1.csv')
print(ad_test.columns)

submission = pd.read_csv('sample_submission.csv')
print(submission.columns)


## Make a result DataFrame
i = 0
colnames = ['model','feat','param', 'auc','train','test']
result = pd.DataFrame(columns=colnames)


## Create features to use a model
feat1 = ['ip_attr_prop','app_attr_prop','device_attr_prop','os_attr_prop','channel_attr_prop','hour_attr_prop','tot_attr_prop']
feat2 = ['ip_hour_prop','ip_app_prop','ip_channel_prop','hour_app_prop','hour_channel_prop','tot_vv_prop']
feat3 = feat1 + feat2
feat4 = ['ip_attr_prop','app_attr_prop','channel_attr_prop','tot_attr_prop']
feat5 = feat4 + feat2
feat6 = ['app_attr_prop','channel_attr_prop','hour_app_prop','hour_channel_prop']


## Predict a target variable
feat = [feat1,feat2,feat3,feat4,feat5,feat6]
name = ['feat1','feat2','feat3','feat4','feat5','feat6']
# sample = '10m_'
# sample = '20m_'
# sample = '30m_'
sample = ''

for f,n in zip(feat,name):
    print("feat = %s" %n)
    
    ## Divid data    
    X_train, X_test, y_train, y_test = train_test_split(ad[f], ad['is_attributed'], random_state=1)
    
    print("X_train : " + str(X_train.shape))
    print("X_test : " + str(X_test.shape))
    print("y_train : " + str(y_train.shape))
    print("y_test : " + str(y_test.shape))
 
    print("y_test : ")
    print(y_test.value_counts())

    
    ## Make a model using Decision Tree    
    for d in [3,4,5]:
        print("When max_depth=%d :" %d)
            
        ## Train the model
        tree = DecisionTreeClassifier(max_depth=d, random_state=1)
        tree.fit(X_train,y_train)
    
        ## predict is_attributed
        p = tree.predict_proba(X_test)[:,1]
        p, train = examine_outlier(p)
    
        ## Evaluate the model
        score = tree.score(X_test,y_test)
        auc = roc_auc_score(y_test, p)
        
        print("feature importance : %s" % tree.feature_importances_)
        print("coefficient of determination : %.5f" % score)
        print("AUC : %.5f" % auc)
    
        ## Predict target variable
        is_attributed = tree.predict_proba(ad_test[f])[:,1]
        is_attributed, test = examine_outlier(is_attributed)
    
        ## Save result
        r = pd.DataFrame({'model':'tree',
                          'param':str(d),
                          'feat':n,
                          'auc':auc,
                          'train':train,
                          'test':test}, columns=colnames, index=[i])
        result = pd.concat([result, r], ignore_index=True)
        i+=1  
    
        submission['is_attributed'] = is_attributed
        submission.to_csv(sample + 'submission1_tree_' + str(d) + '_' + n + '.csv', index=False)
        print("save complete...\n")
    
    
    ## Make a model using Random Forest
    for d in [3,4,5]:
        for e in [30,50,70]:
            for f in [1,2,3]:
                print("When max_depth=%d, n_estimators=%d, max_features=%d :" %(d,e,f))
                
                ## Train the model
                forest = RandomForestClassifier(max_depth=d, n_estimators=e, max_features=f, random_state=1)
                forest.fit(X_train,y_train)
        
                ## predict is_attributed
                p = forest.predict_proba(X_test)[:,1]
                p, train = examine_outlier(p)
                
                ## Evaluate the model
                score = forest.score(X_test,y_test)
                auc = roc_auc_score(y_test, p)
                
                print("feature importance : %s" % forest.feature_importances_)
                print("coefficient of determination : %.5f" % score)
                print("AUC : %.5f" % auc)
            
                ## Predict target variable
                is_attributed = forest.predict_proba(ad_test[f])[:,1]
                is_attributed, test = examine_outlier(is_attributed)
    
                ## Save result
                param = '_'.join(str(x) for x in [d,e,f])
                r = pd.DataFrame({'model':'forest',
                                  'param':param,
                                  'feat':n,
                                  'auc':auc,
                                  'train':train,
                                  'test':test}, columns=colnames, index=[i])
                result = pd.concat([result, r], ignore_index=True)
                i+=1
        
                submission['is_attributed'] = is_attributed
                submission.to_csv(sample + 'submission1_forest_' + param + '_' + n + '.csv', index=False)
                print("save complete...\n")
    
    
    ## Make a model using Gradient Boosting Classifier      
    for d in [3,4,5]:
        for e in [30,50]:
            for l in [0.01,0.1,1,10]:
                print("when max_depth=%d, n_estimators=%d, learning_rate=%.2f : " %(d,e,l))
        
                ## Train the model
                gbrt = GradientBoostingClassifier(max_depth=d, n_estimators=e, learning_rate=l, random_state=1)
                gbrt.fit(X_train,y_train)
    
                ## predict is_attributed
                p = gbrt.predict_proba(X_test)[:,1]
                p, train = examine_outlier(p)
        
                ## Evaluate the model
                score = gbrt.score(X_test,y_test)
                auc = roc_auc_score(y_test, p)
                
                print("feature : %s" % gbrt.feature_importances_)
                print("coefficient of determination : %.5f" % score)
                print("AUC : %.5f" % auc)
        
                ## Predict target variable
                is_attributed = gbrt.predict_proba(ad_test[f])[:,1]
                is_attributed, test = examine_outlier(is_attributed)
    
                ## Save result
                param = '_'.join(str(x) for x in [d,e,l])
                r = pd.DataFrame({'model':'gbrt',
                                  'param':param,
                                  'feat':n,
                                  'auc':auc,
                                  'train':train,
                                  'test':test}, columns=colnames, index=[i])
                result = pd.concat([result, r], ignore_index=True)
                i+=1
    
                submission['is_attributed'] = is_attributed
                submission.to_csv(sample + 'submission1_gbrt_' + param + '_' + n + '.csv', index=False)
                print("save complete...\n")


    ## Save resultset
    result.to_csv(sample + 'result.csv', index=False)