#!/usr/bin/env python
# coding: utf-8

# # AI-Powered Health Assistant Using Patient Data and NLP
# This project develops an AI-based healthcare assistant that leverages Natural Language Processing (NLP) to interact with patients and provide personalized health assessments and prescriptions. The system uses the Hugging Face API to analyze patient-reported symptoms along with health metrics (e.g., temperature, SpO2, heart rate, blood pressure) to generate tailored prescriptions. By accessing patient information via an Aadhar ID and analyzing health data from a comprehensive dataset, the AI delivers recommendations on medications, precautions, and lifestyle changes. This solution enhances patient care by offering quick, accurate health guidance and improving data-driven healthcare accessibility.

# ## Importing necessary libraries

# In[51]:

import numpy as np
import pandas as pd
import csv
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import pyttsx3
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# ## Load Datasets for training and testing
training = pd.read_csv('/home/parallels/Documents/VScode/Symptom-Based-Disease-Prediction-Chatbot-Using-NLP-main/Data/Training.csv')
testing= pd.read_csv('/home/parallels/Documents/VScode/Symptom-Based-Disease-Prediction-Chatbot-Using-NLP-main/Data/Testing.csv')

# Data Pre-processing
cols= training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Mapping categorical strings to numerical labels
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Splitting the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Initialize accuracy lists
train_accuracies_dt = []
val_accuracies_dt = []
train_accuracies_svm = []
val_accuracies_svm = []

# Number of iterations for training
num_iterations = 5

for _ in range(num_iterations):
    # Decision Tree Model
    clf1 = DecisionTreeClassifier()
    clf1.fit(x_train, y_train)
    train_accuracies_dt.append(clf1.score(x_train, y_train))
    val_accuracies_dt.append(clf1.score(x_test, y_test))

    # Support Vector Machine Model
    model = SVC()
    model.fit(x_train, y_train)
    train_accuracies_svm.append(model.score(x_train, y_train))
    val_accuracies_svm.append(model.score(x_test, y_test))

# Print accuracy results
print("Decision Tree Training Accuracies: ", train_accuracies_dt)
print("Decision Tree Validation Accuracies: ", val_accuracies_dt)
print("SVM Training Accuracies: ", train_accuracies_svm)
print("SVM Validation Accuracies: ", val_accuracies_svm)

# Plotting the training and validation accuracy
plt.figure(figsize=(14, 7))

# Training and Validation Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(range(num_iterations), train_accuracies_dt, label='Decision Tree Training Accuracy', marker='o')
plt.plot(range(num_iterations), val_accuracies_dt, label='Decision Tree Validation Accuracy', marker='o')
plt.plot(range(num_iterations), train_accuracies_svm, label='SVM Training Accuracy', marker='o')
plt.plot(range(num_iterations), val_accuracies_svm, label='SVM Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.xticks(range(num_iterations))
plt.legend()
plt.grid()

# Model Accuracy Comparison (Final Training Accuracies)
plt.subplot(1, 2, 2)
models = ['Decision Tree', 'SVM']
final_train_accuracies = [train_accuracies_dt[-1], train_accuracies_svm[-1]]
plt.bar(models, final_train_accuracies, color=['blue', 'orange'])
plt.title('Model Training Accuracy Comparison')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# Remaining functions and logic...

# Initialize dictionaries to store symptom severity, description, and precautions
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

# Dictionary to map symptoms to their indices
symptoms_dict = {}

# Populate symptoms dictionary with indices
for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index

# Function definitions, sensor data, predictions, etc...
# Add your previously defined functions here...

# Call the functions
getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
sensor_data()
tree_to_code(clf1, cols)
print("----------------------------------------------------------------------------------------------------------------------------------")
