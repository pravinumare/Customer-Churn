from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from utils import call_failure_transformer, subscription_length_transformer, seconds_of_use_transformer, frequency_of_use_transformer, frequency_of_SMS_transformer, distinct_called_numbers_transformer, customer_value_transformer, converter



app = Flask(__name__)

rf_model_hyp = pickle.load(open("rf_model_hyp.pkl", "rb"))
columns_list = pickle.load(open("columns_list.obj", 'rb'))



@app.route('/churnprediction')
def churnprediction():

    data = request.get_json()

    call_failure = data['Call Failure']
    complains = data['Complains']
    subscription_length = data['Subscription Length']
    charge_amount = data['Charge Amount']
    seconds_of_use = data['Seconds of Use']
    frequency_of_use = data['Frequency of use']
    frequency_of_SMS = data['Frequency of SMS']
    distinct_called_numbers = data['Distinct Called Numbers']
    age_group = data['Age Group']
    tariff_plan = data['Tariff Plan']
    status = data['Status']
    customer_value = data['Customer Value']

    call_failure = call_failure_transformer(call_failure)
    subscription_length = subscription_length_transformer(subscription_length)
    seconds_of_use = seconds_of_use_transformer(seconds_of_use)
    frequency_of_use = frequency_of_use_transformer(frequency_of_use)
    frequency_of_SMS = frequency_of_SMS_transformer(frequency_of_SMS)
    distinct_called_numbers = distinct_called_numbers_transformer(distinct_called_numbers)
    customer_value = customer_value_transformer(customer_value)
  
    test_df = pd.DataFrame({'Call  Failure':call_failure, 'Complains':complains, 'Subscription  Length':subscription_length,
                        'Charge  Amount':charge_amount, 'Seconds of Use':seconds_of_use, 'Frequency of use':frequency_of_use,
                        'Frequency of SMS':frequency_of_SMS, 'Distinct Called Numbers':distinct_called_numbers,
                        'Age Group':age_group, 'Tariff Plan':tariff_plan, 'Status':status, 'Customer Value':customer_value})

    
    pred = rf_model_hyp.predict(test_df)
    print("\n",">> >> >>  :  ",pred)

    prediction = converter(pred[0])

    return jsonify({"Prediction" : prediction})


if __name__ == "__main__":
    app.run()