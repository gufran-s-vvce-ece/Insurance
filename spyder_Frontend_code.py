# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:52:45 2022

@author: Akash Baskar
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from joblib import dump,load
model=load("insur.joblib")

def main():
    st.title("This is insurance prediction app")
    st.sidebar.title("Prediction")
    age=st.sidebar.number_input("enter your age")
    sex=st.sidebar.selectbox("enter your gender",(0,1))
    bmi=st.sidebar.number_input("enter your BMI")
    child=st.sidebar.number_input("enter No. of childrens")
    smoker=st.sidebar.selectbox("Are you a smoker or not",(0,1))
    if st.sidebar.button("Submit"):
        predictions=model.predict([[age,sex,bmi,child,smoker]])
        st.write(predictions)

if __name__=="__main__":
    main()