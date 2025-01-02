import pickle
from flask import Flask, request, jsonify
import xgboost as xgb

app = Flask("pss score")

with open('train.bin','rb') as f_in :
    dv, model = pickle.load(f_in)

features = list(dv.get_feature_names_out())

@app.route("/predict", methods = ["POST"])
def predict():
    person = request.get_json()

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

    # Applying the Model
    dX = xgb.DMatrix(X, feature_names = features)
    y_pred = model.predict(dX)[0]
    result ={
        "PSS score": float(y_pred)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)


