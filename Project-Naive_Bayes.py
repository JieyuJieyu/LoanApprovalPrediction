#!/usr/bin/env python
# coding: utf-8

# In[25]:


# import libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# import dataset
dataset = pd.read_csv('C:/Users/User/OneDrive/J.E/Sem II 2022_2023/BCI3333 Machine Learning Applications/Final Project/Loan_Train.csv')
dataset = dataset.drop(columns=['Loan_ID'])
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values

# Drop rows with missing values
dataset = dataset.dropna()

# Handling or Encode categorical variables 
labelencoder_X = LabelEncoder()
x[:, 0] = labelencoder_X.fit_transform(x[:, 0])  # Gender
x[:, 1] = labelencoder_X.fit_transform(x[:, 1])  # Married
x[:, 2] = labelencoder_X.fit_transform(x[:, 2])  # Dependents
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])  # Education
x[:, 4] = labelencoder_X.fit_transform(x[:, 4])  # Self_Employed
x[:, 10] = labelencoder_X.fit_transform(x[:, 10])  # Property_Area

# Filling missing values with mean for numeric columns
imputer = SimpleImputer(strategy='mean')
x[:, 5:10] = imputer.fit_transform(x[:, 5:10])

# Fitting Naive Bayes Classifier to the dataset
classifier = GaussianNB()
classifier = classifier.fit(x, y)

# Predict using classifier

# Gender=male, Married=yes, Dependents=2, 
# Education=graduate, Self_Employed=yes, 
# ApplicantIncome=5417, CoapplicantIncome=4196, 
# LoanAmount=267, Loan Amount Term=360, 
# Credit_History=1, Property_Area=Urban
# Loan_Status=? 

prediction = classifier.predict([[1, 1, 2, 0, 1, 5417, 4196, 267, 360, 1, 2]])
print("Prediction:", prediction)

# Predict using the classifier
y_pred = classifier.predict(x)

# Calculate confusion matrix
cm = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Calculate precision
precision = precision_score(y, y_pred, pos_label='Y')
print("Precision:", precision)

# Calculate recall
recall = recall_score(y, y_pred, pos_label='Y')
print("Recall:", recall)

# Calculate F-score
f_score = f1_score(y, y_pred, pos_label='Y')
print("F-Score:", f_score)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)


from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Fitting Naive Bayes Classifier to the training set
classifier = GaussianNB()
classifier = classifier.fit(x_train, y_train)

# Predict on the training set
y_train_pred = classifier.predict(x_train)
training_accuracy = accuracy_score(y_train, y_train_pred)

# Predict on the test set
y_test_pred = classifier.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\nTraining Accuracy:", training_accuracy)
print("Test Accuracy:", test_accuracy)

