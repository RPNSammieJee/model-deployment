# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:08:15 2024

@author: JOSH
"""

import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open("C:/Users/JOSH/.spyder-py3/ml model/trained_model.sav", "rb"))


def diabetes_prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    predict = loaded_model.predict(data)
    if predict == 0:
        return 'This person does not have diabetes'
    else:
        return 'his person has diabetes'

def main():
    st.title("Diabetes Prediction Web App")
    Pregnancies = st.text_input("Enter patient Pregnancies Status: ")
    Glucose = st.text_input("Enter patient Glucose level: ")
    BloodPressure = st.text_input("Enter patient Blood pressure status: ")
    SkinThickness = st.text_input("Enter SkinThickness status: ")
    Insulin = st.text_input("Enter Insulin Status: ")
    BMI = st.text_input("Enter BMI status: ")
    DiabetesPedigreeFunction = st.text_input("Enter DF Diabete: ")
    Age = st.text_input("Enter Patient age: ")
    diagnosis = " "
    
    if st.button("Diabetes test result"):
        diagnosis = diabetes_prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
    st.success(diagnosis)



if __name__ == '__main___':
    main()