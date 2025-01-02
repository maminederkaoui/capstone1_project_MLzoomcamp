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
    X = dv.transform([person])
    dX = xgb.DMatrix(X, feature_names = features)
    y_pred = model.predict(dX)[0]
    result ={
        "PSS score": float(y_pred)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)


