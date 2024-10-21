from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from utils import call_failure_transformer, subscription_length_transformer, seconds_of_use_transformer, frequency_of_use_transformer, frequency_of_SMS_transformer, distinct_called_numbers_transformer, customer_value_transformer, converter


rf_model_hyp = pickle.load(open("rf_model_hyp.pkl", "rb"))
columns_list = pickle.load(open("columns_list.obj", 'rb'))


# Creating the title for the app
st.set_page_config(page_title="Customer Churn Prediction",
                   page_icon='',
                   layout='centered')

# Creating header
st.header("Customer Churn Predictor")

call_failure = st.number_input("Enter the number of call failure")

complains = st.selectbox("Did you raise any complain?",('0', '1'))

subscription_length = st.slider("Select subscription length", 1, 50)

charge_amount = st.slider("Select your charge amount", 0, 10)

seconds_of_use = st.number_input("Enter number of seconds used")

frequency_of_use = st.number_input("Enter frequency of use (within 0 to 250)")

frequency_of_SMS = st.number_input("Enter frequency of SMS (within 0 to 600)")

distinct_called_numbers = st.number_input("Enter distincted call numbers (within 0 to 100)")

age_group = st.selectbox("Enter your Age group",('10-19', '20-29', '30-39', '40-49', '50+'))
if age_group=='10-19':
    age_group=1
elif age_group=='20-29':
    age_group=2
elif age_group=='30-39':
    age_group=3
elif age_group=='40-49':
    age_group=4
else:
    age_group=5

tariff_plan = st.selectbox("Enter your tariff plan", ('1', '2'))

status = st.selectbox("Enter your status", ('1', '2'))

customer_value = st.number_input("Enter customer value")

# Creating button Predict
button = st.button("Predict")

if button:
    
    with st.spinner("Loading please wait..."):
        
        call_failure = call_failure_transformer(call_failure)
        complains = int(complains)
        subscription_length = subscription_length_transformer(subscription_length)
        charge_amount = charge_amount
        seconds_of_use = seconds_of_use_transformer(seconds_of_use)
        frequency_of_use = frequency_of_use_transformer(frequency_of_use)
        frequency_of_SMS = frequency_of_SMS_transformer(frequency_of_SMS)
        distinct_called_numbers = distinct_called_numbers_transformer(distinct_called_numbers)
        age_group = age_group
        tariff_plan = int(tariff_plan)
        status = int(status)
        customer_value = customer_value_transformer(customer_value)

        test_df = pd.DataFrame({'Call  Failure':[call_failure], 'Complains':[complains], 'Subscription  Length':[subscription_length],
                        'Charge  Amount':[charge_amount], 'Seconds of Use':[seconds_of_use], 'Frequency of use':[frequency_of_use],
                        'Frequency of SMS':[frequency_of_SMS], 'Distinct Called Numbers':[distinct_called_numbers],
                        'Age Group':[age_group], 'Tariff Plan':[tariff_plan], 'Status':[status], 'Customer Value':[customer_value]})
        
        print(test_df)

        pred = rf_model_hyp.predict(test_df)
        
        prediction = converter(pred[0])

        print("\nPrediction :",prediction)
        
        st.write(prediction)
