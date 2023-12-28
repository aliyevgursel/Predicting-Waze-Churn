# Predicting-Waze-Churn

**Project Title**

"Predicting Waze User Churn"



**Project Overview**
The overall business goal of this project is to help prevent user churn on the Waze app. This project focuses on monthly user churn. This project will analyze user data and develop a machine-learning (ML) model that predicts user churn. At this stage we constructed and tested couple of ML models: random forest (RF) and extreme gradient boosting (XGB). However, even the 'best' model (which was XGB model) did not predict user churn very well. The recall score with the test data was only 19.5%. The precision was little over 40%, and the accuracy score was about 81%. Therefore there is a need to improve the model.



**Business Understanding**

Approximately 18% of Waze users in the datset churned. The leadership wanted to know: Who are the users most likely to churn? Why do users churn?, and 
When do users churn?



**Data Understanding**

This project used data for 15k users. The dataset had 13 variables like whether user churned or not, how many times a user opened the app during the month, the type of device used (iPhone vs. Android), total kilometers driven in the last month etc. The early descriptive analysis showed that the churn rate is positively correlated with the kilometers driven per day.

![image](https://github.com/aliyevgursel/Predicting-Waze-Churn/assets/68837397/c035ecb7-41d2-4f73-8443-b3bd9cfaafa3)

The data had missing values for the churn status for about 700 users. However, these users did not seem to be different from the rest in terms of the observed characteristics. These users were dropped from the analysis. Moreover, some features were engineered based on the available variables (like kilometers driven per day). 




**Modeling and Evaluation**

The dataset was split into train/validate/test (60%, 20%, 20%) datasets. The models were trained and cross-validated with the train datset. The validation dataset was used to select the "best" model. The 'best' model was the XGB model. However, even this model did not predict user churn very well. Therefore there is a need to improve the model.

![image](https://github.com/aliyevgursel/Predicting-Waze-Churn/assets/68837397/643e2e6d-1722-4fb7-ab84-2f98a0e80d6f)

The engineered features accounted for six of the top 10 features (and three of the top five).

![image](https://github.com/aliyevgursel/Predicting-Waze-Churn/assets/68837397/06031fec-ec0e-46de-a757-3f95fdb0a9dc)

 

**Conclusion**
While the best model can be useful to identify relationships between the churn status and user characteristics, it is not a very good model for the prediction.
Conducting further hyperparameter tuning might improve the model. Increasing sample size might also help. Balancing the data might also help since the dataset is moderately imbalanced. Perhaps we can also add some more features that are likely to be correlated with the churn status. Examples to such features could be a) whether users' primary geographic area is a heavy-trafficked urban area or not; b) car ownership; c) commuting distance etc.
