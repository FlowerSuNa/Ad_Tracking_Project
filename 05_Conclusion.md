##### TalkingData AdTracking Fraud Detection Challenge
# 5. Conclusion

At first, I thought I could tell if it was fraudulent according to ip. So, I made models with features related to ip. But they were overfitting.

<br>

To slove this problem, I've made a blacklist of ip, app, os, device, channel, hour that is many clicks but not many downloads. I extracted click time for each ip and calculated the click gap. And as a result, I made good models without overfitting.


<br>

---

[Contents](README.md) <br>
