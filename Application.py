#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Pass, Selected, Without File Upload Function

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

st.write("""
# Loan Approval Prediction App

This app predicts whether a loan will be approved or not based on user input!

""")

st.sidebar.header('User Input Features')

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    married = st.sidebar.selectbox('Married', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('0', '1', '2', '3+'))
    education = st.sidebar.selectbox('Education', ('Graduate', 'Not Graduate'))
    self_employed = st.sidebar.selectbox('Self Employed', ('Yes', 'No'))
    applicant_income = st.sidebar.slider('Applicant Income', 150, 81000, 5000)
    coapplicant_income = st.sidebar.slider('Coapplicant Income', 0, 41667, 0)
    loan_amount = st.sidebar.slider('Loan Amount', 9, 700, 100)
    loan_amount_term = st.sidebar.slider('Loan Amount Term', 12, 480, 360)
    credit_history = st.sidebar.selectbox('Credit History', ('0', '1'))
    property_area = st.sidebar.selectbox('Property Area', ('Urban', 'Rural', 'Semiurban'))
    data = {'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history,
            'Property_Area': property_area}
    features = pd.DataFrame(data, index=[0])
    return features
    
input_df = user_input_features()

# Displays the user input features
st.subheader('User Input Features')
st.write(input_df)

# Load dataset
dataset = pd.read_csv('C:/Users/User/OneDrive/J.E/Sem II 2022_2023/BCI3333 Machine Learning Applications/Final Project/Loan_Train.csv')
dataset = dataset.drop(columns=['Loan_ID'])
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values

# Drop rows with missing values
dataset = dataset.dropna()

# Preprocess categorical variables
labelencoder_X = LabelEncoder()

# Fit label encoder on all categorical columns in the training data to obtain the unique labels
labelencoder_X.fit(np.concatenate([x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 10]], axis=0))

# Transform the categorical variables in both training and input data
x[:, 0] = labelencoder_X.transform(x[:, 0])  # Gender
x[:, 1] = labelencoder_X.transform(x[:, 1])  # Married
x[:, 2] = labelencoder_X.transform(x[:, 2])  # Dependents
x[:, 3] = labelencoder_X.transform(x[:, 3])  # Education
x[:, 4] = labelencoder_X.transform(x[:, 4])  # Self_Employed
x[:, 10] = labelencoder_X.transform(x[:, 10])  # Property_Area

# Map '3+' to 4 in the Dependents column
x[x[:, 2] == '3+', 2] = '4'

# Convert numerical columns to float
x[:, 5:10] = x[:, 5:10].astype(float)

# Filling missing values with mean for numeric columns
imputer = SimpleImputer(strategy='mean')
x[:, 5:10] = imputer.fit_transform(x[:, 5:10])

# Fitting Naive Bayes Classifier to the dataset
classifier = GaussianNB()
classifier = classifier.fit(x, y)

# Apply model to make predictions
input_data = input_df.values
# Use the same LabelEncoder to transform categorical columns
input_data[:, 0] = labelencoder_X.transform(input_data[:, 0])  # Gender
input_data[:, 1] = labelencoder_X.transform(input_data[:, 1])  # Married
input_data[:, 2] = labelencoder_X.transform(input_data[:, 2])  # Dependents
input_data[:, 3] = labelencoder_X.transform(input_data[:, 3])  # Education
input_data[:, 4] = labelencoder_X.transform(input_data[:, 4])  # Self_Employed
input_data[:, 10] = labelencoder_X.transform(input_data[:, 10])  # Property_Area

input_data[:, 5:10] = imputer.transform(input_data[:, 5:10])

prediction = classifier.predict(input_data)
prediction_proba = classifier.predict_proba(input_data)

# Map labels to integers
label_map = {'N': 0, 'Y': 1}

# Define prediction labels
p_labels = ['Loan Not Approved', 'Loan Approved']

# Convert prediction array to a regular Python list
prediction_list = [label_map[pred] for pred in prediction.tolist()]

st.subheader('Prediction')
st.write(p_labels[prediction_list[0]])

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Calculate confusion matrix
y_pred = classifier.predict(x)
cm = confusion_matrix(y, y_pred)
st.subheader('Confusion Matrix')
st.write(cm)

# Calculate precision
precision = precision_score(y, y_pred, pos_label='Y')
st.subheader('Precision')
st.write(precision)

# Calculate recall
recall = recall_score(y, y_pred, pos_label='Y')
st.subheader('Recall')
st.write(recall)

# Calculate F-score
f_score = f1_score(y, y_pred, pos_label='Y')
st.subheader('F-Score')
st.write(f_score)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
st.subheader('Accuracy')
st.write(accuracy)

