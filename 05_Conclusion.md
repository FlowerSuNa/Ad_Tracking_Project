##### TalkingData AdTracking Fraud Detection Challenge
# 5. Conclusion

At first, I thought I could tell whether it was a fraudulent click by ip. So, I made models with features related to ip. But they were **overfitting**.

<br>

To slove this problem, I've made a blacklist of ip, app, os, device, channel, hour that is many clicks but not many downloads. I extracted click time for each ip and calculated the click gap. And as a result, I made good models without overfitting.

<br>

I thought only features related to ip, app, channel and click time were important, but the performance of the model with features related to device and os was better.

<br>

Performance was different depending on machine learning methods and the features used. The method with the best performance is **LightGBM**.

<br>

---

[Contents](README.md) <br>
