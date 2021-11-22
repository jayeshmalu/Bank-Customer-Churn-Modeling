# Bank Customer Churn Prediction

### Problem Statement : Predicting which customer is going to withdraw his/her account from bank

## Solution :
#### When any customer withdraws his/her account from the bank, that is a big loss for a bank. So to overcome this kind of losses I have built an AI/ML-Based Model that will recommend bank peoples to target those customers who are going to withdraw their account so that they can provide them better service to retain customers


#### About Dataset :

Size of Dataset: 10,000 records

#### FEATURES:

###### CustomerId : Unique id for each customer
###### Surname : Customer surname
###### CreditScore : Credit score based on transaction history
###### Geography : Customer Location
###### Gender : Customer Gender
###### Age : Age of customer
###### Tenure : Period
###### Balance : Customer account balance
###### NumOfProducts : no. of product from banks like credit card / debit card
###### HasCrCard : Customer is having credit card or not
###### IsActiveMember : Customer account is active or not based on frequency of transactions based on recent data
###### EstimatedSalary  : Customer salary
###### Exited : Target varible, i.e customer account withdraw 1- withdraw, 0 - not withdraw

After analysis we encoded some features & dropped some features which are not important

#### Model Building :
So after preparing dataset & Analysis of data, now to accelerate the process of model selection we are used pycaret library to find best model on the basis of Accuracy, AUC, Recall, Precision, F1 Score

Selected model : Gradient Boosting Classifier with Accuracy of 86.6% by tunning some hyperparameter

#### Notebooks:
###### Bank Customer Churn Modeling - Analysis.ipynb  -- Analysis notebook
###### Bank Customer Churn Modeling - Pycaret.ipynb  -- For finding best model
###### Bank Customer Churn Modeling - SK-Learn.ipynb  -- we used sk-learn library to create Gradient Boosting Classifier model
###### Bank Customer Churn Modeling - ANN.ipnb  -- ANN model for prediction - 86% accuracy

#### Model:
churn_model.pkl 
