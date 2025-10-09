#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 20:29:00 2025

@author: vinayakchivilkar
"""

import pickle
import streamlit as st 
from streamlit_option_menu import option_menu
import numpy as np


#loading the saved models

diabetes_model = pickle.load(open('/Users/vinayakchivilkar/Desktop/College/SEM 4/ML_Project/balanced_diabetes_model.sav','rb'))


#heart_disease_model = pickle.load(open('/Users/vinayakchivilkar/Desktop/College/SEM 4/ML_Project/HeartDisease_model.sav','rb'))


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
        

        
        diab_prediction = diabetes_model.predict(input_data)        
        if (diab_prediction[0]==1):
            diab_diagnosis = 'High risk of Diabetes'
        else:
            diab_diagnosis = 'Low risk of Diabetes'
            
    st.success(diab_diagnosis)
    
    
 
    