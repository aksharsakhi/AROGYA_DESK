#!/usr/bin/env python
# coding: utf-8

# # AI-Powered Health Assistant Using Patient Data and NLP
# This project develops an AI-based healthcare assistant that leverages Natural Language Processing (NLP) to interact with patients and provide personalized health assessments and prescriptions. The system uses the Hugging Face API to analyze patient-reported symptoms along with health metrics (e.g., temperature, SpO2, heart rate, blood pressure) to generate tailored prescriptions. By accessing patient information via an Aadhar ID and analyzing health data from a comprehensive dataset, the AI delivers recommendations on medications, precautions, and lifestyle changes. This solution enhances patient care by offering quick, accurate health guidance and improving data-driven healthcare accessibility.

# ## Importing necessary libraries

# In[51]:


# Numpy and pandas for mathematical operations
import numpy as np
import pandas as pd

# To read csv dataset files
import csv

# Regular expression, for pattern matching
import re

# The preprocessing module provides functions for data preprocessing tasks such as scaling and handling missing data.
from sklearn import preprocessing

# For Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# train-test split
from sklearn.model_selection import train_test_split

# For building decision tree models, and _tree to access low-level decision of tree structure
from sklearn.tree import DecisionTreeClassifier, _tree

# For evaluating model performance using cross_validation
from sklearn.model_selection import cross_val_score

# Import Support Vector Classification from sklearn library for model deployment
from sklearn.svm import SVC

# Remove unecessary warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ## Text to speech using pyttsx3

# In[52]:


#sudo apt-get install espeak


# In[53]:


#pip install pyttsx3


# In[54]:


# Import pyttsx3 library
import pyttsx3


# In[55]:


# Initialize the text-to-speech engine
engine = pyttsx3.init()


# In[56]:


# Function to convert text to speech
def text_to_speech(text):
    # Set properties (optional)
    engine.setProperty('rate', 150)    # Speed percent (can go over 100)
    engine.setProperty('volume', 0.9)  # Volume 0-1

    # Convert text to speech
    engine.say(text)
    engine.runAndWait()


# ## Exploratory Data Analysis (EDA)
# 

# In[57]:


# Load Datasets for training and testing
training = pd.read_csv('Data/Training.csv')
testing= pd.read_csv('Data/Testing.csv')


# In[58]:


# Number of rows and columns
shape = training.shape
print("Shape of Training dataset: ", shape)


# In[59]:


# Description about dataset
description = training.describe()
description


# In[60]:


# Information about Dataset
info_df = training.info()
info_df


# In[61]:


# To find total number of null values in dataset
null_values_count = training.isnull().sum()
null_values_count


# In[62]:


# Print First eight rows of the Dataset
training.head(8)


# In[63]:


cols= training.columns
cols= cols[:-1]

# x stores every column data except the last one
x = training[cols]

# y stores the target variable for disease prediction
y = training['prognosis']


# In[64]:


# Figsize used to define size of the figure
plt.figure(figsize=(10, 20))
# Countplot from seaborn on the target varable and data accesed from Training dataset
sns.countplot(y='prognosis', data=training)
# Tile for title of the figur
plt.title('Distribution of Target (Prognosis)')
# Show used to display the figure on screen
plt.show()


# In[65]:


# Grouping Data by Prognosis and Finding Maximum Values
reduced_data = training.groupby(training['prognosis']).max()

# Display the first five rows of the reduced data
reduced_data.head()


# ## Data Pre-processing

# In[66]:


# Mapping categorical strings to numerical labels using LabelEncoder
le = preprocessing.LabelEncoder()

# Fit the label encoder to the target variable 'y' and transform it
le.fit(y)
y = le.transform(y)


# In[67]:


# Splitting the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Features for testing except the last variable
testx    = testing[cols]

# Target variable for Testing
testy    = testing['prognosis']

# Transforming categorical value into numerical labels
testy    = le.transform(testy)


# ## Model building and evaluation

# In[68]:


# Decision Tree Model Implementation
clf1  = DecisionTreeClassifier()

# Fitting the Training Data
clf = clf1.fit(x_train,y_train)

# Cross-Validation for Model Evaluation
scores = cross_val_score(clf, x_test, y_test, cv=3)

# Print the Mean Score
print("Mean Score: ",scores.mean())


# In[69]:


# Creating Support Vector Machine Model
model=SVC()

# Train the model on Training Data
model.fit(x_train,y_train)

# Print accuracy for SVM Model on the training set
print("Accuracy score for svm: ", model.score(x_test,y_test))

# Calculate feature importance using the trained Decision tree classifier
importances = clf.feature_importances_

# Sort indices in descending order based on feature importance
indices = np.argsort(importances)[::-1]

# Get feature names corresponding to their importance score
features = cols


# In[70]:


# Initialize dictionaries to store symptom severity, description, and precautions

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

# Dictionary to map symptoms to their indices
symptoms_dict = {}

# Populate symptoms dictionary with indices
for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

# Function to calculate the overall severity of the symptom
def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")


# Function to read and store symptom descriptions from a CSV file
def getDescription():
    global description_list
    with open('Data/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)



# Function to read and store symptom severity information from a CSV file
def getSeverityDict():
    global severityDictionary
    with open('Data/Symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass



# Function to read and store symptom precaution information from a CSV file
def getprecautionDict():
    global precautionDictionary
    with open('Data/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


# In[71]:


def getInfo():
    print("-----------------------------------AROGYA DESK-----------------------------------")
    print("\nYour Name? \t\t\t\t",end="->")
    name=input("")
    print("Hello", name) 
    print("\nWelcom To AROGYA DESK\n")


# In[72]:


def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]


# In[73]:


def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


# In[74]:


def print_disease(node):
    node = node[0]
    val  = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))


# In[75]:


import pyttsx3
from sklearn.tree import _tree
import pandas as pd

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load medicines data from CSV file
medicines_df = pd.read_csv('Data/medicines.csv')

def get_medicines(disease):
    """Return the list of medicines for the given disease."""
    row = medicines_df[medicines_df['Disease'].str.lower() == disease.lower()]
    if not row.empty:
        medicines = row.iloc[0, 1:].dropna().tolist()  # Get all non-NaN medicines
        return medicines
    return []

def sensor_data():
    print("-----------------------------------SENSOR DATA-----------------------------------\n")
    engine.say("Please enter your temperature in Fahrenheit.")
    engine.runAndWait()
    
    # Input Temperature
    print("\nEnter your Temperature (°F): \t\t", end="-> ")
    temperature = float(input())
    
    engine.say("Please enter your SpO2 level.")
    engine.runAndWait()

    # Input SpO2 level
    print("\nEnter your SpO2 level (%): \t\t", end="-> ")
    spo2 = float(input())
    
    engine.say("Please enter your heart rate in beats per minute.")
    engine.runAndWait()

    # Input Heart Rate
    print("\nEnter your Heart Rate (bpm): \t\t", end="-> ")
    heart_rate = int(input())
    
    engine.say("Please enter your blood pressure in systolic and diastolic format.")
    engine.runAndWait()

    # Input Blood Pressure
    print("\nEnter your Blood Pressure (mmHg) [format: systolic/diastolic]: \t", end="-> ")
    bp = input().split('/')
    systolic = int(bp[0])
    diastolic = int(bp[1])

    # Conditions for Temperature
    if temperature < 95:
        engine.say("Low body temperature. Hypothermia detected. Immediate medical attention needed!")
        print("\nLow body temperature (Hypothermia). Immediate medical attention needed!\n")
    elif 95 <= temperature <= 98.6:
        engine.say("Your body temperature is normal.")
        print("\nNormal body temperature.\n")
    elif 98.7 <= temperature <= 100:
        engine.say("You have a slightly elevated temperature. Monitor your symptoms.")
        print("\nSlightly elevated temperature. Monitor your symptoms.\n")
    elif 100 < temperature <= 102:
        engine.say("You have a fever.")
        print("\nYou have a fever.\n")
    elif temperature > 102:
        engine.say("You have a high fever. Consider consulting a doctor.")
        print("\nHigh fever. Consider consulting a doctor.\n")
    
    # Conditions for SpO2
    if spo2 >= 95:
        engine.say("Your SpO2 level is normal.")
        print("\nNormal SpO2 level.\n")
    elif 90 <= spo2 < 95:
        engine.say("You have mild hypoxia. Monitor your oxygen levels.")
        print("\nMild hypoxia. Monitor your oxygen levels.\n")
    elif spo2 < 90:
        engine.say("Severe hypoxia detected. Immediate medical attention required!")
        print("\nSevere hypoxia. Immediate medical attention required!\n")

    # Conditions for Heart Rate
    if heart_rate < 60:
        engine.say("You have a low heart rate. Consult a doctor.")
        print("\nLow heart rate (Bradycardia). Consult a doctor.\n")
    elif 60 <= heart_rate <= 100:
        engine.say("Your heart rate is normal.")
        print("\nNormal heart rate.\n")
    elif heart_rate > 100:
        engine.say("You have a high heart rate. Monitor your symptoms.")
        print("\nHigh heart rate (Tachycardia). Monitor your symptoms.\n")

    # Conditions for Blood Pressure
    if systolic < 90 or diastolic < 60:
        engine.say("You have low blood pressure.")
        print("\nLow blood pressure (Hypotension).\n")
    elif 90 <= systolic <= 120 and 60 <= diastolic <= 80:
        engine.say("Your blood pressure is normal.")
        print("\nNormal blood pressure.\n")
    elif 120 < systolic <= 140 or 80 < diastolic <= 90:
        engine.say("You have elevated blood pressure.")
        print("\nElevated blood pressure (Prehypertension).\n")
    elif systolic > 140 or diastolic > 90:
        engine.say("You have high blood pressure. Consult a doctor.")
        print("\nHigh blood pressure (Hypertension). Consult a doctor.\n")

    engine.runAndWait()

# Decision Tree logic with Text-to-Speech
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        engine.say("Enter the symptom you are experiencing.")
        engine.runAndWait()
        print("\nEnter the symptom you are experiencing  \t\t", end="->")
        disease_input = input("")

        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            print("searches related to input: ")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            if num != 0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp = 0

            disease_input = cnf_dis[conf_inp]
            break
        else:
            engine.say("Please enter a valid symptom.")
            print("Enter valid symptom.")

    while True:
        try:
            engine.say("How many days have you been experiencing this symptom?")
            print("Okay. From how many days ? : ")
            num_days = int(input())
            break
        except:
            engine.say("Please provide a valid number of days.")
            print("Enter valid input.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])

            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

            engine.say("Are you experiencing any of the following symptoms?")
            engine.runAndWait()
            print("Are you experiencing any ")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = ""
                engine.say(f"{syms}, are you experiencing it?")
                engine.runAndWait()
                print(syms, " ? : ", end='')
                while True:
                    inp = input("")
                    if inp == "yes" or inp == "no":
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ", end="")
                if inp == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)

            # Add medication recommendation feature
            if present_disease[0] == second_prediction[0]:
                engine.say(f"You may have {present_disease[0]}.")
                engine.runAndWait()
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])

                # Get medicines for the predicted disease
                medicines = get_medicines(present_disease[0])
                if medicines:
                    print("\nRecommended Medicines for {}: ".format(present_disease[0]))
                    for medicine in medicines:
                        print(f"- {medicine}")
                    engine.say(f"The recommended medicines for {present_disease[0]} are: {', '.join(medicines)}.")
                    engine.runAndWait()
                else:
                    engine.say("No medicines found for this disease.")
                    engine.runAndWait()
            else:
                engine.say(f"You may have {present_disease[0]} or {second_prediction[0]}.")
                engine.runAndWait()
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

                # Get medicines for the first predicted disease
                medicines = get_medicines(present_disease[0])
                if medicines:
                    print("\nRecommended Medicines for {}: ".format(present_disease[0]))
                    for medicine in medicines:
                        print(f"- {medicine}")
                    engine.say(f"The recommended medicines for {present_disease[0]} are: {', '.join(medicines)}.")
                    engine.runAndWait()
                else:
                    engine.say("No medicines found for the first predicted disease.")
                    engine.runAndWait()

                # Get medicines for the second predicted disease
                medicines = get_medicines(second_prediction[0])
                if medicines:
                    print("\nRecommended Medicines for {}: ".format(second_prediction[0]))
                    for medicine in medicines:
                        print(f"- {medicine}")
                    engine.say(f"The recommended medicines for {second_prediction[0]} are: {', '.join(medicines)}.")
                    engine.runAndWait()
                else:
                    engine.say("No medicines found for the second predicted disease.")
                    engine.runAndWait()

    recurse(0, 1)

# Call the functions
getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
sensor_data()
tree_to_code(clf,cols)
print("----------------------------------------------------------------------------------------------------------------------------------")

