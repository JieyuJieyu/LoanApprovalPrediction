#!/usr/bin/env python
# coding: utf-8

# In[906]:


import pandas as pd
import numpy as np

# Load the dataset
url = 'https://raw.githubusercontent.com/JieyuJieyu/LoanApprovalPrediction/main/Loan_Train.csv'
dataset = pd.read_csv(url)
dataset.info()


# In[907]:


print(dataset.describe())


# In[908]:


#Identify the unique values & counts for each categorical column
categorical_columns = dataset.select_dtypes(include=['object']) 
for column in categorical_columns:
    print("Column:", column)
    print(dataset[column].value_counts())
    print()


# In[909]:


dataset = dataset.drop(columns=['Loan_ID'])
categorical_columns = ['Gender', 
                       'Married', 
                       'Dependents',
                       'Education',
                       'Self_Employed',
                       'Property_Area', 
                       'Credit_History',
                       'Loan_Amount_Term']
numerical_columns = ['ApplicantIncome',
                     'CoapplicantIncome',
                     'LoanAmount']


# In[910]:


# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

fig,axes = plt.subplots(4,2,figsize=(12,15))
for idx,cat_col in enumerate(categorical_columns):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=dataset, 
                  hue='Loan_Status',ax=axes[row,col])

plt.subplots_adjust(hspace=1)


# In[911]:


fig,axes = plt.subplots(1,3,figsize=(17,5))
for idx,cat_col in enumerate(numerical_columns):
    sns.boxplot(y=cat_col,data=dataset,
                x='Loan_Status',ax=axes[idx])

print(dataset[numerical_columns].describe())
plt.subplots_adjust(hspace=1)


# In[912]:


# Identify missing values

missing_values = dataset.isnull().sum()
print(missing_values)


# In[913]:


# This will remove all rows with missing values.
dataset = dataset.dropna()
dataset


# In[914]:


# Identify data types
data_types = dataset.dtypes
print(data_types)


# In[915]:


# Model 1: Decision Tree Classification


# In[916]:


# Define X, Y
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,11].values


# In[917]:


# Handling or Encode categorical variables 
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
x[:,0] = labelencoder_X.fit_transform(x[:,0])
x[:,1] = labelencoder_X.fit_transform(x[:,1])
x[:,2] = labelencoder_X.fit_transform(x[:,2])
x[:,3] = labelencoder_X.fit_transform(x[:,3])
x[:,4] = labelencoder_X.fit_transform(x[:,4])
x[:,10] = labelencoder_X.fit_transform(x[:,10])


# In[918]:


# Step 2 - Decision Tree classification

#splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.59,random_state = 0)


# In[919]:


# Step 3 - Decision Tree Classification

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)


# In[920]:


# Step 4 - Decision Tree Classification

# Fitting Decision Tree Classification classifer to the training set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)


# In[921]:


# Step 5 - Decision Tree Classification

# Predict the test set results
y_pred = classifier.predict(x_test)
y_test, y_pred


# In[922]:


# Step 6 - Decision Tree Classification Performance Evaluation 

# Import necessary libraries
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate precision
precision = precision_score(y_test, y_pred, pos_label='Y')
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred, pos_label='Y')
print("Recall:", recall)

# Calculate F-score
f_score = f1_score(y_test, y_pred, pos_label='Y')
print("F-Score:", f_score)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[923]:


y_train_pred = classifier.predict(x_train)
training_accuracy = accuracy_score(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_pred)

print("Training Accuracy:", training_accuracy)
print("Test Accuracy:", test_accuracy)


# In[924]:


# Get feature importances
feature_importances = classifier.feature_importances_

# Create a DataFrame to store feature importances
importance_df = pd.DataFrame({'Feature': dataset.columns[:-1], 'Importance': feature_importances})

# Sort the features by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Select the top k features (e.g., top 8 features)
k = 8
selected_features = importance_df['Feature'].head(k).tolist()

# Print the selected features
print("Selected Features:")
print(selected_features)


# In[925]:


# Decision Tree Classification - Prediction

# Gender=male, Married=yes, Dependents=2, 
# Education=graduate, Self_Employed=no, 
# ApplicantIncome=3200, CoapplicantIncome=700, 
# LoanAmount=70, Loan Amount Term=360, 
# Credit_History=1,Property_Area=Urban
# Loan_Status=?   
y_pred_new = classifier.predict(sc_x.transform(np.array([[0, 1, 2, 0, 0, 3200, 700, 70, 360, 1, 2]])))
print(y_pred_new)


# In[926]:


# Model 2: Naive Bayes 


# In[927]:


# Define X, Y
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,11].values


# In[928]:


# Handling or Encode categorical variables 
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
x[:,0] = labelencoder_X.fit_transform(x[:,0])
x[:,1] = labelencoder_X.fit_transform(x[:,1])
x[:,2] = labelencoder_X.fit_transform(x[:,2])
x[:,3] = labelencoder_X.fit_transform(x[:,3])
x[:,4] = labelencoder_X.fit_transform(x[:,4])
x[:,10] = labelencoder_X.fit_transform(x[:,10])


# In[929]:


# Fitting Naive Bayes Classifier to the dataset

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier = classifier.fit(x,y)


# In[930]:


# Predict using classifier

# Gender=male, Married=yes, Dependents=2, 
# Education=graduate, Self_Employed=no, 
# ApplicantIncome=3200, CoapplicantIncome=700, 
# LoanAmount=70, Loan Amount Term=360, 
# Credit_History=1,Property_Area=Urban
# Loan_Status=?  
prediction = classifier.predict([[0, 1, 2, 0, 0, 3200, 700, 70, 360, 1, 2]])
print(prediction)


# In[933]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Predict using the classifier
y_pred = classifier.predict(x)

# Calculate confusion matrix
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
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


# In[934]:


from sklearn.feature_selection import SelectKBest, chi2

# Perform feature selection using chi-square test
selector = SelectKBest(score_func=chi2, k=8)  # Select top 8 features
x_selected = selector.fit_transform(x, y)

# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)

# Get the names of the selected features
selected_features = dataset.columns[selected_feature_indices]

# Print the selected feature names
print("Selected Features:")
print(selected_features)


# In[935]:


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

print("Training Accuracy:", training_accuracy)
print("Test Accuracy:", test_accuracy)


# In[936]:


# Define custom mappings for categorical variables
gender_mapping = {0: 'Male', 1: 'Female'}
married_mapping = {0: 'No', 1: 'Yes'}
dependents_mapping = {0: '0', 1: '1', 2: '2', 3: '3'}
education_mapping = {0: 'Graduate', 1: 'Not Graduate'}
self_employed_mapping = {0: 'No', 1: 'Yes'}
property_area_mapping = {0: 'Rural', 1: 'Semiurban', 2: 'Urban'}

# Create a dictionary to store the mappings
mappings = {
    'Gender': gender_mapping,
    'Married': married_mapping,
    'Dependents': dependents_mapping,
    'Education': education_mapping,
    'Self_Employed': self_employed_mapping,
    'Property_Area': property_area_mapping
}

# Print the mappings
for feature, mapping in mappings.items():
    print(f"Mapping for {feature}:")
    for encoded_value, original_value in mapping.items():
        print(f"{original_value} -> {encoded_value}")
    print()

