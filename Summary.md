##### TalkingData AdTracking Fraud Detection Challenge
# Summary

* [EDA](#EDA)

* [Preprocessing](#decision-tree)

* [Sampling](#random-forest)

* [Modeling](#gradient-boosting)

* [Conclusion](#lightgbm)

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

[Page Up](#Summary) <br>
[More Details](01_EDA.md) <br>
[Contents](README.md) <br>

---

## Preprocessing

#### Made Features

* gap : Click Count - Download Count

* rate : Download Count / Click Count

* black : 1 if gap is big and rate is low

* click_gap : Click Gap per ip

<br>

[Page Up](#Summary) <br>
[More Details](02_Preprocessing.md) <br>
[Contents](README.md) <br>

---

## Sampling

Extract the most recent data because it is time series data.

<br>

[Page Up](#Summary) <br>
[More Details](03_Sampling.md) <br>
[Contents](README.md) <br>

---

## Modeling



<br>

[Page Up](#Summary) <br>
[More Details](04_Modeling.md) <br>
[Contents](README.md) <br>

---

## Conclusion

At first, I thought I could tell whether it was a fraudulent click by ip, appand channel. So, I made rate features of ip, app, device, os, channel and hour and then, made models with the rate features. But it bacame **overfitting** by feature related on ip.

<br>

To slove this problem, I've made a blacklist of ip, app, os, device, channel, hour that is many clicks but not many downloads. I extracted click time for each ip and calculated the click gap. And as a result, I made **good models** without overfitting.

<br>

I thought only features related to ip, app, channel and click time were important, but the performance of the model with features related to device and os was **better**.


<br>

[Page Up](#Summary) <br>
[Contents](README.md) <br>

---
