# ArogyaDeskAI - Symptom-Based Disease Prediction Chatbot

## Overview

ArogyaDeskAI is an AI-powered health diagnosis chatbot that uses machine learning algorithms to interpret user-reported symptoms and provide health insights. It guides patients in identifying potential medical conditions and offers personalized recommendations for treatment and precautions. Designed for user-friendly interaction, this system aims to make healthcare more accessible and efficient.

## Table of Contents

- Description
- Features
- Usage
- Dataset
- Model Architecture
- Future Work
- Troubleshooting
- Contact Info

## Description

This project implements ArogyaDeskAI, a healthcare chatbot for disease prediction based on patient symptoms. The system leverages machine learning algorithms like Decision Trees and Support Vector Classification (SVC) to predict diseases. It can analyze the user's reported symptoms and health metrics (e.g., temperature, SpO2, heart rate, blood pressure) and recommend a course of treatment, including medication and precautions.

## Features

1. **Symptom Analysis**: Users can input their symptoms, and the chatbot analyzes the input to predict potential diseases.
2. **Health Metrics Integration**: The chatbot also takes additional health data like temperature, SpO2, heart rate, and blood pressure for more accurate predictions.
3. **Recommendations**: It provides medication suggestions, treatment duration, precautions, and foods to avoid based on the patient's condition.
4. **User-Friendly Interface**: The chatbot is designed for easy interaction and guidance.
5. **Data Storage**: All generated prescriptions and patient data are stored securely for future analysis and updates.

## Usage

1. Start the chatbot application.
2. The user will provide their Aadhar ID, and the chatbot will retrieve basic details (name, age, gender, phone number) from `aadhardata.csv`.
3. The chatbot greets the user and asks about their symptoms and health metrics.
4. Follow the chatbot’s instructions to provide details such as temperature, SpO2, heart rate, and blood pressure.
5. Receive disease predictions and treatment recommendations, which are stored in a database for future use.

## Dataset

The project uses the following datasets to make predictions:
- `symptom_Description.csv`: Contains descriptions of symptoms and their severity.
- `symptom_precaution.csv`: Provides precautions for the identified symptoms.
- `Symptom_severity.csv`: Specifies the severity level of symptoms.
- `Training.csv` & `Testing.csv`: Datasets used for model training and evaluation.
- `medicines.csv`: Datasets of medicines for predected disease.

These datasets are stored locally within the project folder and help the chatbot analyze patient inputs to provide reliable results.

## Model Architecture

The disease detection model is built using machine learning algorithms, specifically Decision Trees and Support Vector Classification (SVC). The model is trained on symptom-disease mappings and health metrics, enabling accurate predictions based on user input.

## Future Work

1. **Improved Accuracy**: Further development of the chatbot to include more complex symptom-disease mappings and historical data.
2. **User History**: Tracking patient history for personalized recommendations and long-term analysis.
3. **Web or Mobile Application**: Deploying the chatbot as a web or mobile application for broader accessibility.
4. **Integration of New Datasets**: Continuously update datasets to cover more diseases and health conditions.

## Troubleshooting

If you encounter issues while running the chatbot:

1. Ensure all necessary Python libraries and dependencies are installed.
2. Verify that the datasets are correctly formatted and accessible within the project directory.
3. Check that input/output data is properly passed to the chatbot system.

## Contact Information

For any inquiries or contributions, please reach out to:

**Name**: Akshar Sakhi  
**Email**: aksharsakhi@gmail.com 
**All rights reserved by Team MAN-C**
