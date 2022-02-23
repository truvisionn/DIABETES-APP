

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image



st.write("""
## DIABETICS STATUS PREDICTION APP!
""")

image=Image.open('diabetes.jpg')
st.image(image,caption='Diabetes Kills')
model = pickle.load(open('class_model.pkl', 'rb'))

scaler=pickle.load(open('scaler2.pkl', 'rb'))

st.sidebar.header('User Input Parameters')



def user_input_features():
    GenHlth = st.selectbox('Health Rating',('Excellent', 'Good', 'Stable', 'Bad', 'Critical'))
    if GenHlth=='Excellent':
        GenHlth=5
    if GenHlth=='Good':
        GenHlth=4
    if GenHlth=='Stable':
        GenHlth=3
    if GenHlth=='Bad':
        GenHlth=2
    else:
        GenHlth=1
    CholCheck = st.selectbox('Cholestrol Level',('High', 'Low'))
    if CholCheck=='High':
        CholCheck=1
    else:
        CholCheck=0
        
    HighBP = st.selectbox('Blood Pressure Level',('High', 'Low'))
    if HighBP=='High':
        HighBP=1
    else:
        HighBP=0
        
    AnyHealthcare = st.selectbox('Do you Have a Health Care Plan',('Yes','No'))
    if AnyHealthcare=='Yes':
        AnyHealthcare=1
    else:
        AnyHealthcare=0
   
    PhysActivity = st.selectbox('Do You engage in Regular Physical Activity', ('Yes', 'No'))
    if PhysActivity=='Yes':
        PhysActivity=1
    else:
        PhysActivity=0
        

    Veggies=st.selectbox('Do You Take Vegetables',('Yes','No'))
    if Veggies=='Yes':
        Veggies=1
    else:
        Veggies=0
        
    data = {'GenHlth':GenHlth,
           'CholCheck':CholCheck,
           'HighBP':HighBP,
           'AnyHealthcare':AnyHealthcare,
           'PhysActivity':PhysActivity,
           'Veggies':Veggies}
    
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()
input_df = scaler.transform(input_df)


if st.button('PREDICT'):
    y_out=model.predict(input_df)
    if y_out[0]==1:
        st.write(f' You have a high risk of Diabeties')
    else:
        st.write(f' You are not at risk of Diabetes')
   
    


