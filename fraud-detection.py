import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("./datasets/log.csv")
print(data.head(3))

#checking if data has nay null values 
print(data.isnull().sum())
#It does not have any null values 

#looking for the different type of transactions in Type column 
print(data.type.value_counts())
#creating a pie/dougnut chart 

type = data.type.value_counts()
transactions = type.index
quantity = type.values 

fig, ax = plt.subplots()
ax.pie(quantity, labels = transactions, autopct = '%1.1f%%')
plt.title("Distribution of Transaction Type")
plt.show()

#Looking for correlation between fatures of data with isFraud column 
correlation = data.corr(numeric_only = True)
print(correlation["isFraud"].sort_values())

#converting the categorical data into numerical so that it can used to train a model 
data["type"] = data["type"].map({
   "CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4,"DEBIT": 5 
})
#converting the target labels into categorical labels. 
data["isFraud"] = data["isFraud"].map({
    0: "No Fraud" , 1: "Fraud"
})
print(data.head(3))

#splitting the dataset into two parts : train and test sets 
x = data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]]
y = data["isFraud"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state=1)

#Logistic Regression model 
log_model = LogisticRegression().fit(x_train, y_train)
log_model_predict = log_model.predict(x_test)
log_model_accuracy = accuracy_score(y_test, log_model_predict)
print("The accuracy for logistic regression model is :", log_model_accuracy)

#Decision Tree Classifier 
tree_model = DecisionTreeClassifier().fit(x_train, y_train)
tree_model_predict = tree_model.predict(x_test)
tree_model_accuracy = accuracy_score(y_test, tree_model_predict)
print("The accuracy for Decision tree model is :", tree_model_accuracy)
