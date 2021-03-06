# LibSeek

LibSeek is an open-source prototype software that can be used for recommending potential useful third-party libraries for mobile app development. The most significant characteristic of it is that it can not only effectively give out the recommendation list, but also diversify the prediction results, which can bring more benefits for developers.

- LibSeek employs an adaptive weighting mechanism to neutralize the bias caused by the popularity of third-party libraries. 
- It introduces neighborhood information, i.e., information about similar apps and similar third-party libraries, to personalize the predictions for individual apps.

Please feel free to use this dataset in your research, and it would be appreciated if you could cite the following paper.

> Diversified Third-Party Library Prediction for Mobile App Development. Q. He, B. Li, F. Chen, J. Grundy, X. Xia and Y. Yang. IEEE Transactions on Software Engineering (2020).

# How to use

This prototype is made via Matlab, thus please download the source script and open it with Matlab. It has been tested under Matlab 2017b, however, older or newer versions should be fine to execute it.

The dataset used in LibSeek can be found at [MALib](https://github.com/malibdata/MALib-Dataset). Please put both the dataset and the source file in the current folder of your Matlab before execute it.
