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

This section should detail what models you used and the corresponding evaluation metrics. 

**Conclusion**
In the conclusion section explain the recommendations you have in solving the business problem and highlight any future steps you will take to expand on your project
