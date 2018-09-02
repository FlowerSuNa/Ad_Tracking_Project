##### TalkingData AdTracking Fraud Detection Challenge
# 5. Conclusion

At first, I thought I could tell whether or not this is a fraudulent click by ip, app and channel. So, I made the download rate per features(ip, app, device, os, channel and hour) and then, made models with this features. But it bacame **overfitting** by the download rate per ip.

<br>

To slove this problem, I've made blacklist of ip, app, os, device, channel, hour that is many clicks but not many downloads. I also extracted click time for each ip and calculated the click gap. As a result, I made **good models** without overfitting.

<br>

I thought only features related to ip, app, channel and click time were important, but the performance of the model with features related to device and os was better.

<br>

Performance was different depending on machine learning methods and the features used. The method with the best performance is **LightGBM**. The features used are rate_app, rate_os, rate_channel, gap_app, gap_channel, black_ip, black_device, click_gap.

<br>

---

[Contents](README.md) <br>
