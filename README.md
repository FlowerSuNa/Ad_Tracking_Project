# Kaggle - TalkingData AdTracking Fraud Detection Challenge

TalkingData, China’s largest independent big data service platform, covers over 70% of active mobile devices nationwide. They handle 3 billion clicks per day, of which 90% are potentially fraudulent. The goal of the competition is to create an algorithm that predicts whether a user will download an app after clicking a mobile app ad.

---

## Summary
- Summary : [pdf](TalkingData%20AdTracking.pdf) <br>
- Evaluation : [xlsx](performance_evaluation.xlsx) <br>

---

## Contents Table
♣ : source code

1. Data Exporation [♣](01_Data_Exporation.py) <br>
&ensp; - Check the number of downloads over time : [timeplot](graph/sample_timeplot.png) <br>
&ensp; - Check click count, download count, download rate <br>
&emsp;&emsp;&emsp;&emsp;&emsp; per hour : [barplot](graph/sample_barplot_hour.png) <br>
&emsp;&emsp;&emsp;&emsp;&emsp; by app : [barplot](graph/sample_barplot_app.png) <br>
&emsp;&emsp;&emsp;&emsp;&emsp; by device : [barplot](graph/sample_barplot_device.png) <br>
&emsp;&emsp;&emsp;&emsp;&emsp; by os : [barplot](graph/sample_barplot_os.png) <br>
&emsp;&emsp;&emsp;&emsp;&emsp; by channel : [barplot](graph/sample_barplot_channel.png) <br>
&ensp; - Check correlation : [scatterplot](graph/sample_scatterplot.png) <br>

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
