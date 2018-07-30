# Kaggle - TalkingData AdTracking Fraud Detection Challenge

#### Description
TalkingData, China’s largest independent big data service platform, covers over 70% of active mobile device
s nationwide. They handle 3 billion clicks per day, of which 90% are potentially fraudulent. The goal of the
competition is <b>to create an algorithm that predicts whether a user will download an app after clicking a
mobile app ad</b>.

#### Evalution
Submissions are evaluated on <b>area under the ROC curve</b> between the predicted probability and the
observed target.

#### Variable
* ip : ip address of click
* app : app id for marketing
* device : device type id of user mobile phone
* os : os version id of user mobile phone
* channel : channel id of mobile ad publisher
* click_time : timestamp of click (UTC)
* attributed_time : if user download the app for after clicking an ad, this is the time of the app download
* is_attributed : the target that is to be predicted, indicating the app was download

---

## Summary
- Summary : [pdf](TalkingData%20AdTracking.pdf) <br>
- Evaluation : [xlsx](performance_evaluation.xlsx) <br>

---

## Contents Table
♣ : source code <br>
★ : view

1. EDA [♣](01_EDA.py) [★](01_EDA.md) <br>
2. Preprocessing
3. Modeling

---

### Method1
2. Data Preprocessing <br>
&ensp; - Train Data Preprocessing [♣](method1_02_1_Train_Data_Preprocessing.py) <br>
&ensp; - Test Data Preprocessing [♣](method1_02_2_Test_Data_Preprocessing.py) <br>
3. Target Variable Prediction <br>
&ensp; - Linear Regression, Ridge, Logistic Regression [♣](method1_03_1_Target_Variable_Prediction.py) <br>
&ensp; - Decision Tree, Random Forest, Gradient Boosting [♣](method1_03_2_Target_Variable_Prediction.py) <br>
&ensp; - K-Nearest Neighbors, Support Vector Machines [♣](method1_03_3_Target_Variable_Prediction.py) <br>
&ensp; - LightGBM [♣](method1_03_4_Target_Variable_Prediction.py) <br>

### Method2
2. Data Preprocessing [♣](method2_02_Data_Preprocessing.py) <br>
3. Target Variable Prediction <br>
&ensp; - LightGBM [♣](method2_03_Target_Variable_Prediction.py) <br>

### Method3
2. Data Preprocessing [♣](method3_02_Data_Preprocessing.py) <br>
3. Target Variable Prediction <br>
&ensp; - LightGBM [♣](method3_03_Target_Variable_Prediction.py) <br>
