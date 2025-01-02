import requests

url = "http://localhost:9696/predict"

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

result = requests.post(url, json=person).json()

print(result)