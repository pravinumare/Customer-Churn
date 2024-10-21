import numpy as np
from scipy.stats import boxcox


def call_failure_transformer(call_failure):
    
    call_failure = np.cbrt(call_failure)

    return call_failure

def subscription_length_transformer(sl):
    
    boxcox_lambda_val_Subscription_Length = 2.4543572516155967
    New = ((sl**boxcox_lambda_val_Subscription_Length) - 1) / boxcox_lambda_val_Subscription_Length
    
    return New

def seconds_of_use_transformer(seconds_of_use):
    
    seconds_of_use = np.sqrt(seconds_of_use)

    return seconds_of_use

def frequency_of_use_transformer(frequency_of_use):
    
    frequency_of_use = np.sqrt(frequency_of_use)

    return frequency_of_use

def frequency_of_SMS_transformer(frequency_of_SMS):
    
    frequency_of_SMS = np.cbrt(frequency_of_SMS)

    return frequency_of_SMS

def distinct_called_numbers_transformer(distinct_called_numbers):
    
    distinct_called_numbers = np.sqrt(distinct_called_numbers)

    return distinct_called_numbers

def customer_value_transformer(customer_value):
    
    customer_value = np.sqrt(customer_value)

    return customer_value

def converter(pred):

    if pred == 0:
        return "Non-Churn"
    else:
        return "Churn Possible"