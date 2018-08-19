##### TalkingData AdTracking Fraud Detection Challenge
# Summery

* [EDA](#EDA)

* [Decision Tree](#decision-tree)

* [Random Forest](#random-forest)

* [Gradient Boosting](#gradient-boosting)

* [LightGBM](#lightgbm)

* [Trial and error](trial/Trial.md)

---

## EDA

#### Data Shape

| Data | Col | Row |
|:----:|:---:|----:|
| Train | 8 | 184,903,890 |
| Test | 7 | 18,790,469 |

> The data is very large!!!

<br>

#### Missing Values

* Train

| ip | app | device | os | channel | click_time | attributed_time | is_attributed |
|:--:|:---:|:------:|:--:|:-------:|:----------:|:---------------:|:-------------:|
| 0 | 0 | 0 | 0 | 0 | 0 | 184,447,004 | 0 |

> attributed_time value is NaN if is_attributed value is 0.

<br>

* Test

| click_id | ip | app | device | os | channel | click_time |
|:--------:|:--:|:---:|:------:|:--:|:-------:|:----------:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |

<br>

#### Level Size of Feature

| Data | ip | app | device | os | channel |
|:----:|---:|----:|-------:|---:|--------:|
| Train | 277,396 | 406 | 3,475 | 800 | 202 |
| Test | 93,936 | 417 | 1,985 | 395 | 178 |

> There are too many levels.

<br>

#### Download Frequency

| Target | Count |
|:------:|------:|
| Not Downloaded | 18,447,044 |
| Downloaded | 456,846 |

> Have very few downloads.

<br>

[Page Up](#Summery) <br>
[more details](01_EDA.md) <br>
[Contents](README.md) <br>

---

## Preprocessing

#### Made Features

* gap : Click Count - Download Count (per ip, app, device, os, channel, hour)

* rate : Download Count / Click Count (per ip, app, device, os, channel, hour)

* black : 1 if gap is big and rate is low (per ip, app, device, os, channel, hour)

* click_gap : The gap of next click per ip

<br>

[Page Up](#Summery) <br>
[more details](02_Preprocessing.md) <br>
[Contents](README.md) <br>

---

## Sampling

Extract the most recent data because it is time series data.

<br>

[Page Up](#Summery) <br>
[more details](03_Sampling.md) <br>
[Contents](README.md) <br>

---

## Modeling

Extract the most recent data because it is time series data.

<br>

[Page Up](#Summery) <br>
[more details](04_Modeling.md) <br>
[Contents](README.md) <br>

---
