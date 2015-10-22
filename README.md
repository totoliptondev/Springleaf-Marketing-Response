# Springleaf-Marketing-Response
The Springleaf-Marketing-Response script is for the Kaggle challenge listed on https://www.kaggle.com/c/springleaf-marketing-response.

Springleaf is a personal loan company, and the goal of this challenge is to use their customer data predict whether or not a potential customer will be use their service. The training dataset is 149.83Mb zipped and 921Mb unzipped, and the test dataset is 149.94Mb zipped and 921Mb unzipped. Each training sample has 1933 features and 1 binary label. Each test sample has 1933 features only. There are 145231 training samples and 145232 test samples.  

The springleaf_script.r provided in this repository reads in the .csv data, converts all the data into numerics, replace missing samples with -1, and finally use xgboost package to predict the probability in the test set. The result are evaluted using area under the ROC curve between the predicted probability and the observed target, the highest score obtained by this script is 0.77495. The highest score in the competition is 0.80427, and the score obtained by this script is around 50% percentile of all submissions.

# Things that can be improved on:
1. The training set were subsamples due to the 4GB memory limitation of my machine. Simply by using all training set to train the xgboost model would increase the score.   
2. Tweaking the model parameters such as the max_depth, eta, subsample, colsample_bytree and etc would result in a higher score. This is a search problem in the parameter space and there are special tools designed to do this. 
