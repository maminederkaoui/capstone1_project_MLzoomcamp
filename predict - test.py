import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
import pickle

with open('train.bin','rb') as f_in :
    dv, model = pickle.load(f_in)

person = {
    'openness' : 3,
    'conscientiousness': 3.5, 
    'extraversion': 1.5,
    'agreeableness': 3, 
    'neuroticism': 4, 
    'sleep_time': 7, 
    'wake_time' : 8,
    'sleep_duration': 7, 
    'psqi_score': 2, 
    'call_duration': 0.5, 
    'num_calls': 2, 
    'num_sms': 40,
    'screen_on_time': 2, 
    'skin_conductance': 1, 
    'accelerometer': 1,
    'mobility_radius': 1.5, 
    'mobility_distance': 6
}

features = list(dv.get_feature_names_out())

# Fields to keep
columns_to_keep = [
    'psqi_score', 'conscientiousness', 'extraversion', 'neuroticism', 
    'sleep_time', 'sleep_duration', 'num_calls', 'num_sms', 
    'skin_conductance', 'mobility_distance'
]

# Lowercasing keys and filtering the dictionary
person = {key: value for key, value in person.items() if key.lower() in columns_to_keep}

# Tranforming person's data
X = dv.transform([person])

dX = xgb.DMatrix(X, feature_names = features)
y_pred = model.predict(dX)[0]

print(f"The PSS score of the person is {y_pred}")

