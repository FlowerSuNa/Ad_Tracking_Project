##### TalkingData AdTracking Fraud Detection Challenge
# 4. Modeling
[source code](04_Modeling_Logistic.py)

<br>

---

## Train Size : 20,000,000

| Dataset | Not Downloaded | Downloaded | Total |
|:-------:|---------------:|-----------:|------:|
| Train | 15,962,214 | 37,786 | 16,000,000
| Valid | 3,990,607 | 9,393 | 4,000,000 |

<br>

#### Features :
#### black_ip, black_app, black_os, black_channel

Score

| Model | Parameter | value | Train AUC | Valid AUC | Score |
|:-----:|:---------:|:-----:|:---------:|:---------:|:-----:|
| Logistic Regression | C | 0.01 | 0.91760 | 0.91851 | - |
|  |  | 0.1 | 0.91760 | 0.91851 | - |
|  |  | 1 | 0.91760 | 0.91851 | - |
|  |  | 10 | 0.91760 | 0.91851 | - |
|  |  | 100 | 0.91760 | 0.91851 | 0.9050371 |
| Decision Tree | max_depth | 1 | 0.83560 | 0.83797 | - |
|  |  | 2 | 0.89017 | 0.89141 | - |
|  |  | 3 | 0.91725 | 0.91811 | - |
|  |  | 4 | 0.91823 | 0.91906 | - |
|  |  | 5 | 0.91823 | 0.91906 | 0.9056274 |
| Random Forest | max_depth | 1 , 50 , 1 |  |  | - |
|  | n_estimators | 1 , 50 , 2 |  |  | - |
|  | max_features | 1 , 50 , 3 |  |  | - |

Coefficient

| Model | Parameter | value | black_ip | black_app | black_os | black_channel |
|:-----:|:---------:|:-----:|:--------:|:---------:|:--------:|:-------------:|
| Logistic Regression | C | 0.01 | -3.06639 | -3.62914 | 0.35751 | -1.86023 |
|  |  | 0.1 | -3.12880 | -3.72217 | 0.43949 | -1.86185 |
|  |  | 1 | -3.13530 | -3.73185 | 0.44802 | -1.86196 |
|  |  | 10 | -3.13585 | -3.73261 | 0.44865 | -1.86179|
|  |  | 100 | -3.13592 | -3.73270 | 0.44874 | -1.86179 |

Feature Importance

| Model | Parameter | value | black_ip | black_app | black_os | black_channel |
|:-----:|:---------:|:-----:|:--------:|:---------:|:--------:|:-------------:|
| Decision Tree | max_depth | 1 | 0 | 1 | 0 | 0 |
|  |  | 2 | 0.57799 | 0.42201 | 0 | 0 |
|  |  | 3 | 0.49437 | 0.36096 | 0.04157 | 0.10901 |
|  |  | 4 | 0.48144 | 0.35151 | 0.06401 | 0.10303 |
|  |  | 5 | 0.48144 | 0.35151 | 0.06401 | 0.10303 |

<br>

#### Features :
#### black_ip, gap_app, black_app, gap_os, black_os, gap_channel, black_channel

| Parameter : C | Train AUC | Valid AUC | Score |
|:-------------:|:---------:|:---------:|:-----:|
| 1 | 0.90897 | 0.90805 | - |

<br>

#### Features :
#### black_ip, gap_app, black_app, gap_device, gap_os, black_os, gap_channel,
#### black_channel, black_hour
