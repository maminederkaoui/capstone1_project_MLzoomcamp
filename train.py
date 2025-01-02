import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
import pickle

df = pd.read_csv("data/stress_detection.csv")

# Keeping only the features having more than a 1% correlation score with PSS_score
columns_to_keep = ['PSS_score', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'sleep_time', 'sleep_duration', 'PSQI_score', 'num_calls', 'num_sms', 'skin_conductance', 'mobility_distance']
df = df[columns_to_keep]

# Renaming columns
df.columns = df.columns.str.lower()

df_full_train, df_test = train_test_split (df, test_size = 0.20, random_state = 10)
y_full_train = df_full_train["pss_score"]
y_test = df_test["pss_score"]
del df_full_train["pss_score"]
del df_test["pss_score"]

dv = DictVectorizer(sparse = False)

dict_full_train = df_full_train.to_dict(orient = "records")
dict_test = df_test.to_dict(orient = "records")

X_full_train = dv.fit_transform(dict_full_train)
X_test = dv.transform(dict_test)
features = list(dv.get_feature_names_out())

dfull_train = xgb.DMatrix(X_full_train, label = y_full_train, feature_names = features)
dtest = xgb.DMatrix(X_test, label = y_test, feature_names = features)
watchlist = [(dfull_train, 'full_train'), (dtest, 'test')]

xgb_params = {
    'eta': 0.05, 
    'max_depth': 10,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}
model = xgb.train(xgb_params, dfull_train, num_boost_round= 11,
                evals=watchlist)

y_pred = model.predict(dtest)

rmse_score = root_mean_squared_error(y_test, y_pred)
print(f"RMSE score on the test dataset is {rmse_score}")

with open('train.bin', 'wb') as f_out :
    pickle.dump((dv, model), f_out)


