#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 20:12:45 2025

@author: vinayakchivilkar
"""

import pickle
import streamlit as st 
from streamlit_option_menu import option_menu
import numpy as np


#loading the saved models

diabetes_model = pickle.load(open('/Users/vinayakchivilkar/Desktop/College/SEM 4/ML_Project/Diabetes_model.sav','rb'))
scaler = pickle.load(open('/Users/vinayakchivilkar/Desktop/College/SEM 4/ML_Project/scaler.sav', 'rb'))

heart_disease_model = pickle.load(open('/Users/vinayakchivilkar/Desktop/College/SEM 4/ML_Project/HeartDisease_model.sav','rb'))


#sidebar
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction'],
                           icons = ['activity',
                                    'heart'],
                           default_index = 0)


if (selected == 'Diabetes Prediction'):
    
    st.title('Diabetes Prediction using ML')
    
    
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
        
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
        
    with col2:
        Insulin = st.text_input('Insulin level')

    with col3:
        BMI = st.text_input('BMI Value')
        
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
        
    with col2:
        Age = st.text_input('Age of the person')

    #code for prediction
    diab_diagnosis = ''
    
    #button for Prediction
    
    if st.button('Diabetes Test Result'):
        
        input_data = np.array([
            float(Pregnancies), float(Glucose), float(BloodPressure), 
            float(SkinThickness), float(Insulin), float(BMI), 
            float(DiabetesPedigreeFunction), float(Age)
        ]).reshape(1, -1)
        
        input_data_scaled = scaler.transform(input_data)
        
        diab_prediction = diabetes_model.predict(input_data_scaled)        
        if (diab_prediction[0]==1):
            diab_diagnosis = 'High risk of Diabetes'
        else:
            diab_diagnosis = 'Low risk of Diabetes'
            
    st.success(diab_diagnosis)
    
    
    
if (selected == 'Heart Disease Prediction'):
    st.title('Heart Disease Prediction using ML')
    
    
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.text_input('Age of the Person')
        
    with col2:
        Sex = st.text_input('Gender')

    with col3:
        cp = st.text_input('Chest Pain Types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestrol in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results (values 0,1,2)')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    #code for prediction
    heart_diagnosis = ''
    
    
    #button for Prediction 
    if st.button('Heart Disease Test Result'):
        input_data = np.array([Age, Sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal], dtype=float)
        heart_prediction = heart_disease_model.predict([input_data])
        
        if (heart_prediction[0]==1):
            heart_diagnosis = 'High risk of Heart Disease'
        else:
            heart_diagnosis = 'Low risk of Heart Disease'
            
    st.success(heart_diagnosis)
    