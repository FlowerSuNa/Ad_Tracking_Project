##### TalkingData AdTracking Fraud Detection Challenge
# 4. Modeling
[source code](04_Modeling.py) <br>

Find the important features and then, make the model using Logistic Regression, Decision Tree, Random Forest, Gradient boosting and LightGBM.

<br>

* [Logistic Regression](#logistic-regression)

* [Decision Tree](#decision-tree)

* [Random Forest](#random-forest)

* [Gradient Boosting](#gradient-boosting)

* [LightGBM](#lightgbm)

* [Trial and error](trial/Trial.md)

---

## Split Train Data

#### Train Size : 10,000,000

| Dataset | Not Downloaded | Downloaded | Total |
|:-------:|---------------:|-----------:|------:|
| Train | 7,982,502 | 17,498 | 8,000,000 |
| Valid | 1,995,618 | 4,382 | 2,000,000 |

<br>

#### Train Size : 20,000,000

| Dataset | Not Downloaded | Downloaded | Total |
|:-------:|---------------:|-----------:|------:|
| Train | 15,962,214 | 37,786 | 16,000,000 |
| Valid | 3,990,607 | 9,393 | 4,000,000 |

<br>

#### Train Size : 30,000,000

| Dataset | Not Downloaded | Downloaded | Total |
|:-------:|---------------:|-----------:|------:|
| Train | 23,939,416 | 60,584 | 24,000,000 |
| Valid | 5,984,690 | 15,310 | 6,000,000 |

<br>

#### Train Size : 40,000,000

| Dataset | Not Downloaded | Downloaded | Total |
|:-------:|---------------:|-----------:|------:|
| Train | 31,919,562 | 80,438 | 32,000,000 |
| Valid | 7,979,930 | 20,070 | 8,000,000 |

<br>

#### Train Size : 50,000,000

| Dataset | Not Downloaded | Downloaded | Total |
|:-------:|---------------:|-----------:|------:|
| Train | 39,897,655 | 102,345 | 40,000,000 |
| Valid | 9,974,300 | 25,700 | 10,000,000 |

<br>

#### Train Size : All

| Dataset | Not Downloaded | Downloaded | Total |
|:-------:|---------------:|-----------:|------:|
| Train | 147,557,584 | 365,528 | 147,923,112 |
| Valid | 36,889,460 | 91,318 | 36,980,778 |

<br>

---

## Logistic Regression

* Train Size : 20,000,000

<br>

#### Used Features : black_ip, black_app, black_os, black_channel

* Score

| C | Train AUC | Valid AUC | Score |
|:-:|:---------:|:---------:|:-----:|
| 0.01 | 0.91760 | 0.91851 | - |
| 0.1 | 0.91760 | 0.91851 | - |
| 1 | 0.91760 | 0.91851 | - |
| 10 | 0.91760 | 0.91851 | - |
| 100 | 0.91760 | 0.91851 | 0.9050371 |

<br>

* Coefficient

| C | black_ip | black_app | black_os | black_channel |
|:-:|:--------:|:---------:|:--------:|:-------------:|
| 0.01 | -3.06639 | -3.62914 | 0.35751 | -1.86023 |
| 0.1 | -3.12880 | -3.72217 | 0.43949 | -1.86185 |
| 1 | -3.13530 | -3.73185 | 0.44802 | -1.86196 |
| 10 | -3.13585 | -3.73261 | 0.44865 | -1.86179|
| 100 | -3.13592 | -3.73270 | 0.44874 | -1.86179 |

<br>

#### Used Features : black_ip, gap_app, black_app, gap_os, black_os, gap_channel, black_channel

* Score

| C | Train AUC | Valid AUC | Score |
|:-:|:---------:|:---------:|:-----:|
| 0.1 | 0.90897 | 0.90805 | - |
| 1 | 0.90897 | 0.90805 | 0.9003460 |
| 10 | 0.90897 | 0.90805 | - |

<br>

* Coefficient

| C | black_ip | gap_app | black_app | gap_os | black_os | gap_channel | black_channel |
|:-:|:--------:|:-------:|:---------:|:------:|:--------:|:-----------:|:-------------:|
| 0.1 | -1.06494834e-11 | -3.95847172e-07 | -9.31501883e-12 | -7.60239423e-08 | -8.53923308e-12 | -2.20993191e-06 | -5.93182256e-12 |
| 1 | -1.06494834e-11 | -3.95847172e-07 | -9.31501883e-12 | -7.60239423e-08 | -8.53923308e-12 | -2.20993191e-06 | -5.93182256e-12 |
| 10 | -1.06494834e-11 | -3.95847172e-07 | -9.31501883e-12 | -7.60239423e-08 | -8.53923308e-12 | -2.20993191e-06 | -5.93182256e-12 |

<br>

#### Used Features : black_ip, gap_app, black_app, gap_os, black_os, gap_channel, black_channel, black_hour, click_gap

* Score

| C | Train AUC | Valid AUC | Score |
|:-:|:---------:|:---------:|:-----:|
| 1 | 0.91057 | 0.90964 | 0.9008474 |

<br>

* Coefficient

| C | black_ip | gap_app | black_app | gap_os | black_os | gap_channel | black_channel | black_hour | click_gap |
|:-:|:--------:|:-------:|:---------:|:------:|:--------:|:-----------:|:-------------:|:----------:|:---------:|
| 1 | -7.33339065e-08 | -3.96425129e-07 | -6.51432349e-08 | -7.63216968e-08 | -5.99720133e-08 | -2.24701394e-06 | -3.70434353e-08 | -2.96075221e-08 | 6.05839932e-05 |

<br>

#### Used Features : black_ip, gap_app, black_app, gap_os, gap_channel, black_channel, click_gap

* Score

| C | Train AUC | Valid AUC | Score |
|:-:|:---------:|:---------:|:-----:|
| 1 | 0.91057 | 0.90964 | 0.9008474 |

<br>

* Coefficient

| C | black_ip | gap_app | black_app | gap_os | gap_channel | black_channel | click_gap |
|:-:|:--------:|:-------:|:---------:|:------:|:-----------:|:-------------:|:---------:|
| 1 | -7.33338368e-08 | -3.96425136e-07 | -6.51431731e-08 | -7.63216985e-08 | -2.24701394e-06 | -3.70434000e-08 |  6.05839333e-05 |

<br>

#### Used Features : black_ip, gap_app, black_app, gap_device, gap_os, gap_channel, black_channel, click_gap

* Score

| C | Train AUC | Valid AUC | Score |
|:-:|:---------:|:---------:|:-----:|
| 1 | 0.91271 | 0.91225 | 0.9021244 |

<br>

* Coefficient

| C | black_ip | gap_app | black_app | gap_device | gap_os | gap_channel | black_channel | click_gap |
|:-:|:--------:|:-------:|:---------:|:----------:|:------:|:-----------:|:-------------:|:---------:|
| 1 | -5.08335955e-12 | -2.50819503e-07 | -3.62569221e-12 | -1.26183283e-08 | -2.07392698e-08 | -1.65163236e-06 | -3.09966527e-12 |  8.14621067e-09 |

<br>

[Page Up](#4-modeling)

<br>

---

## Decision Tree

* Train Size : 20,000,000
* Parameter : max_depth

<br>

#### Used Features : black_ip, black_app, black_os, black_channel

* Score

| max_depth | Train AUC | Valid AUC | Score |
|:---------:|:---------:|:---------:|:-----:|
| 1 | 0.83560 | 0.83797 | - |
| 2 | 0.89017 | 0.89141 | - |
| 3 | 0.91725 | 0.91811 | - |
| 4 | 0.91823 | 0.91906 | - |
| 5 | 0.91823 | 0.91906 | 0.9056274 |

<br>

* Feature Importance

| max_depth | black_ip | black_app | black_os | black_channel |
|:---------:|:--------:|:---------:|:--------:|:-------------:|
| 1 | 0 | 1 | 0 | 0 |
| 2 | 0.57799 | 0.42201 | 0 | 0 |
| 3 | 0.49437 | 0.36096 | 0.04157 | 0.10901 |
| 4 | 0.48144 | 0.35151 | 0.06401 | 0.10303 |
| 5 | 0.48144 | 0.35151 | 0.06401 | 0.10303 |

<br>

#### Used Features : black_ip, gap_app, black_app, gap_os, black_os, gap_channel, black_channel

* Score

| max_depth | Train AUC | Valid AUC | Score |
|:---------:|:---------:|:---------:|:-----:|
| 3 | 0.93339  | 0.93385 | - |
| 4 | 0.93505 | 0.93548 | - |
| 5 | 0.94870 | 0.94847 | - |
| 6 | 0.95446 | 0.95382 | - |
| 7 | 0.95878 | 0.95779 | **0.9455029** |

<br>

* Feature Importance

| max_depth | black_ip | gap_app | black_app | gap_os | black_os | gap_channel | black_channel |
|:---------:|:--------:|:-------:|:---------:|:------:|:--------:|:-----------:|:-------------:|
| 3 | 0.45970 | 0.20465 | 0.33564 | 0 | 0 | 0 | 0 |
| 4 | 0.41370 | 0.26571 | 0.30206 | 0 | 0 | 0.00248 | 0.01605 |
| 5 | 0.38732 | 0.27541 | 0.28280 | 0.00703 | 0 | 0.02476 | 0.02267 |
| 6 | 0.37239 | 0.26694 | 0.27189 | 0.02149 | 0.00563 | 0.03721 | 0.02445 |
| 7 | 0.36299 | 0.26704 | 0.26503 | 0.02663 | 0.00899 | 0.04477 | 0.00245 |

<br>

#### Used Features : black_ip, gap_app, black_app, gap_os, black_os, gap_channel, black_channel, black_hour, click_gap

* Score

| max_depth | Train AUC | Valid AUC | Score |
|:---------:|:---------:|:---------:|:-----:|
| 3 | 0.89129 | 0.89260 | - |
| 4 | 0.93382 | 0.93433 | - |
| 5 | 0.93528 | 0.93575 | - |
| 6 | 0.94879 | 0.94859 | - |
| 7 | 0.95459 | 0.95373 | 0.9385287 |

<br>

* Feature Importance

| max_depth | black_ip | gap_app | black_app | gap_os | black_os | gap_channel | black_channel | black_hour | click_gap |
|:---------:|:--------:|:-------:|:---------:|:------:|:--------:|:-----------:|:-------------:|:----------:|:---------:|
| 3 | 0.46734 | 0.11181 | 0.35044 | 0 | 0 | 0 | 0 | 0 | 0.07041 |
| 4 | 0.41448 | 0.18752 | 0.31080 | 0 | 0 | 0.00024 | 0.01545 | 0 | 0.07150 |
| 5 | 0.38186 | 0.22978 | 0.28634 | 0.00000 | 0 | 0.00301 | 0.01682 | 0 | 0.08218 |
| 6 | 0.35960 | 0.25318 | 0.26965 | 0.00732 | 0 | 0.00398 | 0.01780 | 0 | 0.08848 |
| 7 | 0.34442 | 0.24543 | 0.25823 | 0.00718 | 0 | 0.03184 | 0.02312 | 0.00002 | 0.08922 |

<br>

#### Used Features : black_ip, gap_app, black_app, gap_os, gap_channel, black_channel, click_gap

* Score

| max_depth | Train AUC | Valid AUC | Score |
|:---------:|:---------:|:---------:|:-----:|
| 3 | 0.89129 | 0.89260 | - |
| 4 | 0.93382 | 0.93433 | - |
| 5 | 0.93528 | 0.93575 | - |
| 6 | 0.94879 | 0.94859 | - |
| 7 | 0.95459 | 0.95373 | 0.9385287 |

<br>

* Feature Importance

| max_depth | black_ip | gap_app | black_app | gap_os | gap_channel | black_channel | click_gap |
|:---------:|:--------:|:-------:|:---------:|:------:|:-----------:|:-------------:|:---------:|
| 3 | 0.46734 | 0.11181 | 0.35044 | 0 | 0 | 0 | 0.07041 |
| 4 | 0.41448 | 0.18752 | 0.31080 | 0 | 0.00024 | 0.01545 | 0.07150 |
| 5 | 0.38189 | 0.22978 | 0.28634 | 0.00000 | 0.00301 | 0.01682 | 0.08218 |
| 6 | 0.35960 | 0.25318 | 0.26964 | 0.00731 | 0.00398 | 0.01780 | 0.08848 |
| 7 | 0.34442 | 0.24543 | 0.25823 | 0.00772 | 0.03184 | 0.02312 | 0.08924 |

<br>

#### Used Features : black_ip, gap_app, black_app, gap_device, gap_os, gap_channel, black_channel, click_gap

* Score

| max_depth | Train AUC | Valid AUC | Score |
|:---------:|:---------:|:---------:|:-----:|
| 3 | 0.89128 | 0.89260 | - |
| 4 | 0.93381 | 0.93432 | - |
| 5 | 0.93528 | 0.93575 | - |
| 6 | 0.94879 | 0.94859 | - |
| 7 | 0.95461 | 0.95385 | 0.9386715 |

<br>

* Feature Importance

| max_depth | black_ip | gap_app | black_app | gap_device | gap_os | gap_channel | black_channel | click_gap |
|:---------:|:--------:|:-------:|:---------:|:----------:|:------:|:-----------:|:-------------:|:---------:|
| 3 | 0.46545 | 0.03722 | 0.34902 | 0.07818 | 0 | 0 | 0 | 0.07013 |
| 4 | 0.41540 | 0.10699 | 0.31149 | 0.07974 | 0 | 0.00025 | 0.01548 | 0.07065 |
| 5 | 0.38415 | 0.15930 | 0.28806 | 0.07374 | 0.00000 | 0.00303 | 0.01694 | 0.07477 |
| 6 | 0.36049 | 0.19161 | 0.27032 | 0.06920 | 0.00607 | 0.00399 | 0.01659 | 0.08174 |
| 7 | 0.34342 | 0.18514 | 0.25748 | 0.07268 | 0.00644 | 0.02919 | 0.02267 | 0.08297 |

<br>

[Page Up](#4-modeling)

<br>

---

## LightGBM

* Parameter

| Parameter | Value | Describe |
|-----------|-------|----------|
| boosting_type | gbdt |  |
| objective | binary |  |
| min_split_gain | 0 | lambda_l1, lambda_l2 and min_gain_to_split to regularization |
| reg_alpha | 0 | L1 regularization term on weights |
| reg_lambda | 0 | L2 regularization term on weights |
| nthread | 4 |  |
| verbose | 0 |  |
| metric | auc |  |
| learning_rate | 0.15 |  |
| num_leaves | 7 | 2^max_depth - 1 |
| max_depth | 3 | -1 means no limit |
| min_child_samples | 100 | Minimum number of data need in a child(min_data_in_leaf) |
| max_bin | 100 | Number of bucketed bin for feature values |
| subsample | 0.7 | Subsample ratio of the training instance |
| subsample_freq | 1 | frequence of subsample, <=0 means no enable |
| colsample_bytree | 0.9 | Subsample ratio of columns when constructing each tree |
| min_child_weight | 0 | Minimum sum of instance weight(hessian) needed in a child(leaf) |
| scale_pos_weight | 99 |  |

<br>

#### Used Features : black_ip, black_app, black_os, black_channel

* Score

| Train Size | Train AUC | Valid AUC | Score |
|:----------:|:---------:|:---------:|:-----:|
| 20m | 0.91823 | 0.91906 | 0.9056274 |
| 30m | 0.91139 | 0.91109 | - |
| 40m | 0.90582 | 0.90802 | - |
| 50m | 0.90242 | 0.90179 | - |
| All | 0.90933 | 0.90831 | - |

<br>

* Feature Importance

| Train Size | black_ip | black_app | black_os | black_channel |
|:----------:|:--------:|:---------:|:--------:|:-------------:|
| 20m | 0.13975 | 0.55410 | 0.03579 | 0.27036 |
| 30m | 0.13559 | 0.54250 | 0.03857 | 0.28334 |
| 40m | 0.13661 | 0.53292 | 0.03534 | 0.29512 |
| 50m | 0.41667 | 0.25000 | 0.25000 | 0.08333 |
| All | 0.23077 | 0.23077 | 0.30770 | 0.23077 |

<br>

#### Used Features : black_ip, gap_app, black_app, gap_os, black_os, gap_channel, black_channel

* Score

| Train Size | Train AUC | Valid AUC | Score |
|:----------:|:---------:|:---------:|:-----:|
| 20m | 0.97367 | 0.97143 | **0.9631122** |
| 30m | 0.96925 | 0.96665 | - |
| 40m | 0.96663 | 0.96556 | - |
| 50m | 0.95140 | 0.95293 | - |
| All | 0.95741 | 0.95687 | - |

<br>

* Feature Importance

| Train Size | black_ip | gap_app | black_app | gap_os | black_os | gap_channel | black_channel |
|:----------:|:--------:|:-------:|:---------:|:------:|:--------:|:-----------:|:-------------:|
| 20m | 0.07101 | 0.62613 | 0.24900 | 0.02153 | 0.00099 | 0.03133 |
| 30m | 0.07418 | 0.62355 | 0.21729 | 0.01996 | 0.00097 | 0.03401 | 0.03003 |
| 40m | 0.07980 | 0.62537 | 0.21135 | 0.02283 | 0.00143 | 0.02983 | 0.02940 |
| 50m | 0.11765 | 0.29412 | 0.17647 | 0.17647 | 0 | 0.17647 | 0.05882 |
| All | 0.08181 | 0.45946 | 0.10811 | 0.18919 | 0.00000 | 0.10811 | 0.05405 |

<br>

#### Used Features : black_ip, gap_app, black_app, gap_os, black_os, gap_channel, black_channel, black_hour, click_gap

* Score

| Train Size | Train AUC | Valid AUC | Score |
|:----------:|:---------:|:---------:|:-----:|
| 20m | 0.97505 | 0.97258 | **0.9640591** |
| 30m | 0.97099 | 0.96780 | - |
| 40m | 0.96855 | 0.96681 | - |
| 50m | 0.96382 | 0.96425 | - |
| All | 0.96926 | 0.96839 | - |

<br>

* Feature Importance

| Train Size | black_ip | gap_app | black_app | gap_os | black_os | gap_channel | black_channel | black_hour | click_gap |
|:----------:|:--------:|:-------:|:---------:|:------:|:--------:|:-----------:|:-------------:|:----------:|:---------:|
| 20m | 0.08407 | 0.47933 | 0.34617 | 0.02043 | 0.00082 | 0.02808 | 0.02266 | 0.00047 | 0.01796 |
| 30m | 0.08624 | 0.47479 | 0.33419 | 0.02038 | 0.00097 | 0.03233 | 0.02859 | 0.00059 | 0.02193 |
| 40m | 0.09298 | 0.47531 | 0.33028 | 0.02098 | 0.00079 | 0.02859 | 0.03027 | 0.00046 | 0.02034 |
| 50m | 0.26316 | 0.31579 | 0.10526 | 0.10526 | 0.00000 | 0.15789 | 0.00000 | 0.00000 | 0.05263 |
| All | 0.44000 | 0.12000 | 0.16000 | 0.16000 | 0.00000 | 0.12000 | 0.00000 | 0.00000 | 0.00000 |

<br>

#### Used Features : black_ip, gap_app, black_app, gap_os, gap_channel, black_channel, click_gap

* Score

| Train Size | Train AUC | Valid AUC | Score |
|:----------:|:---------:|:---------:|:-----:|
| 20m | 0.97500 | 0.97238 | **0.9641844** |
| 30m | 0.97054 | 0.96756 | - |
| 40m | 0.96839 | 0.96667 | - |
| 50m | 0.96246 | 0.96297 | 0.9582971 |
| All | 0.96901 | 0.96817 | - |

<br>

* Feature Importance

| Train Size | black_ip | gap_app | black_app | gap_os | gap_channel | black_channel | click_gap |
|:----------:|:--------:|:-------:|:---------:|:------:|:-----------:|:-------------:|:---------:|
| 20m | 0.06776 | 0.61353 | 0.22739 | 0.02072 | 0.02777 | 0.02243 | 0.02039 |
| 30m | 0.06954 | 0.61264 | 0.21882 | 0.02016 | 0.03120 | 0.02560 | 0.02205 |
| 40m | 0.07388 | 0.61509 | 0.20939 | 0.02254 | 0.02997 | 0.02697 | 0.02236 |
| 50m | 0.19444 | 0.22222 | 0.11111 | 0.11111 | 0.13889 | 0.02778 | 0.19444 |
| All | 0.20000 | 0.36000 | 0.20000 | 0.04000 | 0.12000 | 0.08000 | 0.00000 |

<br>

#### Used Features : black_ip, gap_app, black_app, gap_device, gap_os, gap_channel, black_channel, click_gap

* Score

| Train Size | Train AUC | Valid AUC | Score |
|:----------:|:---------:|:---------:|:-----:|
| 20m | 0.97554 | 0.97309 | **0.9651695** |
| 30m | 0.97123 | 0.96854 | - |
| 40m | 0.96329 | 0.96208 | - |
| 50m | 0.93738 | 0.93842 | - |
| All | 0.95467 | 0.95406 | - |

<br>

* Feature Importance

| Train Size | black_ip | gap_app | black_app | gap_device | gap_os | black_os | gap_channel | black_channel |
|:----------:|:--------:|:-------:|:---------:|:----------:|:------:|:--------:|:-----------:|:-------------:|
| 20m | 0.06770 | 0.61012 | 0.22993 | 0.00537 | 0.01901 | 0.02882 | 0.02148 | 0.01757 |
| 30m | 0.06950 | 0.60699 | 0.22441 | 0.00591 | 0.01988 | 0.02842 | 0.02470 | 0.02020 |
| 40m | 0.19355 | 0.35484 | 0.09677 | 0.19355 | 0.03226 | 0.09677 | 0.03226 | 0.00000 |
| 50m | 0.18519 | 0.33333 | 0.18519 | 0.14815 | 0.00000 | 0.11111 | 0.03737 | 0.00000 |
| All | 0.18182 | 0.36364 | 0.22727 | 0.00000 | 0.04545 | 0.13636 | 0.04545 | 0.00000 |

<br>

[Page Up](#4-modeling)

<br>

---

## Random Forest

* Train Size : 10,000,000
* Parameter : max_depth, n_estimators, max_features

#### Used Features : black_ip, gap_app, black_app, gap_device, gap_os, gap_channel, black_channel, click_gap

* Score

| max_depth | n_estimators | max_features | Train AUC | Valid AUC | Score |
|:---------:|:------------:|:------------:|:---------:|:---------:|:-----:|
| 3 | 50 | 3 | 0.95652 | 0.95715 | - |
|  |  | 4 | 0.95764 | 0.95818 | - |
|  |  | 5 | 0.95406 | 0.95427 | - |
|  | 70 | 3 | 0.95653 | 0.95709 | - |
|  |  | 4 | 0.95520 | 0.95583 | - |
|  |  | 5 |  |  | - |
|  | 100 | 3 |  |  | - |
|  |  | 4 |  |  | - |
|  |  | 5 |  |  | - |
| 4 | 50 | 3 |  |  | - |
|  |  | 4 |  |  | - |
|  |  | 5 |  |  | - |
|  | 70 | 3 |  |  | - |
|  |  | 4 |  |  | - |
|  |  | 5 |  |  | - |
|  | 100 | 3 |  |  | - |
|  |  | 4 |  |  | - |
|  |  | 5 |  |  | - |
| 5 | 50 | 3 |  |  | - |
|  |  | 4 |  |  | - |
|  |  | 5 |  |  | - |
|  | 70 | 3 |  |  | - |
|  |  | 4 |  |  | - |
|  |  | 5 |  |  | - |
|  | 100 | 3 |  |  | - |
|  |  | 4 |  |  | - |
|  |  | 5 |  |  | - |

<br>

* Feature Importance

| max_depth | n_estimators | max_features | black_ip | gap_app | black_app | gap_device | gap_os | black_os | gap_channel | black_channel |
|:---------:|:------------:|:------------:|:--------:|:-------:|:---------:|:----------:|:------:|:--------:|:-----------:|:-------------:|
| 3 | 50 | 3 | 0.34546 | 0.15024 | 0.27932 | 0.03129 | 0.00449 | 0.02699 | 0.04542 | 0.11679 |
|  |  | 4 | 0.32877 | 0.18268 | 0.29605 | 0.02414 | 0.00621 | 0.01067 | 0.04571 | 0.10577 |
|  |  | 5 | 0.39212 | 0.16930 | 0.27666 | 0.02987 | 0.00190 | 0.00873 | 0.00683 | 0.11460 |
|  | 70 | 3 | 0.35778 | 0.13942 | 0.27657 | 0.02678 | 0.00738 | 0.03485 | 0.04100 | 0.11622 |
|  |  | 4 | 0.33220 |
|  |  | 5 |  |  | - |
|  | 100 | 3 |  |  | - |
|  |  | 4 |  |  | - |
|  |  | 5 |  |  | - |
| 4 | 50 | 3 |  |  | - |
|  |  | 4 |  |  | - |
|  |  | 5 |  |  | - |
|  | 70 | 3 |  |  | - |
|  |  | 4 |  |  | - |
|  |  | 5 |  |  | - |
|  | 100 | 3 |  |  | - |
|  |  | 4 |  |  | - |
|  |  | 5 |  |  | - |
| 5 | 50 | 3 |  |  | - |
|  |  | 4 |  |  | - |
|  |  | 5 |  |  | - |
|  | 70 | 3 |  |  | - |
|  |  | 4 |  |  | - |
|  |  | 5 |  |  | - |
|  | 100 | 3 |  |  | - |
|  |  | 4 |  |  | - |
|  |  | 5 |  |  | - |

<br>

[Page Up](#4-modeling)

<br>

---

## Gradient Boosting

* Train Size : 10,000,000
* Parameter : max_depth, n_estimators, learning_rate

#### Used Features : black_ip, gap_app, black_app, gap_device, gap_os, gap_channel, black_channel, click_gap

* Score

| max_depth | n_estimators | learning_rate | Train AUC | Valid AUC | Score |
|:---------:|:------------:|:-------------:|:---------:|:---------:|:-----:|
| 3 | 30 | 0.1 |
|  |  | 0.01 |
|  |  | 0.001 |
|  | 50 | 0.1 |  |  | - |
|  |  | 0.01 |
|  |  | 0.001 |
|  | 70 | 0.1 |  |  | - |
|  |  | 0.01 |
|  |  | 0.001 |
| 4 | 30 | 0.1 |
|  |  | 0.01 |
|  |  | 0.001 |
|  | 50 | 0.1 |  |  | - |
|  |  | 0.01 |
|  |  | 0.001 |
|  | 70 | 0.1 |  |  | - |
|  |  | 0.01 |
|  |  | 0.001 |
| 5 | 30 | 0.1 |
|  |  | 0.01 |
|  |  | 0.001 |
|  | 50 | 0.1 |  |  | - |
|  |  | 0.01 |
|  |  | 0.001 |
|  | 70 | 0.1 |  |  | - |
|  |  | 0.01 |
|  |  | 0.001 |

<br>

* Feature Importance

| max_depth | n_estimators | learning_rate | black_ip | gap_app | black_app | gap_device | gap_os | black_os | gap_channel | black_channel |
|:---------:|:------------:|:-------------:|:--------:|:-------:|:---------:|:----------:|:------:|:--------:|:-----------:|:-------------:|
| 3 | 30 | 0.1 |
|  |  | 0.01 |
|  |  | 0.001 |
|  | 50 | 0.1 |  |  | - |
|  |  | 0.01 |
|  |  | 0.001 |
|  | 70 | 0.1 |  |  | - |
|  |  | 0.01 |
|  |  | 0.001 |
| 4 | 30 | 0.1 |
|  |  | 0.01 |
|  |  | 0.001 |
|  | 50 | 0.1 |  |  | - |
|  |  | 0.01 |
|  |  | 0.001 |
|  | 70 | 0.1 |  |  | - |
|  |  | 0.01 |
|  |  | 0.001 |
| 5 | 30 | 0.1 |
|  |  | 0.01 |
|  |  | 0.001 |
|  | 50 | 0.1 |  |  | - |
|  |  | 0.01 |
|  |  | 0.001 |
|  | 70 | 0.1 |  |  | - |
|  |  | 0.01 |
|  |  | 0.001 |

<br>

[Page Up](#4-modeling)

<br>

---

[Contents](README.md) <br>
[3. Sampling](03_Sampling.md) <br>
[5. Conclusion](05_Conclusion.md)
