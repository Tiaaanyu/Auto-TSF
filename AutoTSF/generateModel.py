from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
X = np.loadtxt('meta_data.txt', dtype=np.float32, delimiter=',', usecols=range(1, 39))
where_are_nan = np.isnan(X)
where_are_inf = np.isinf(X)
X[where_are_nan] = 0
X[where_are_inf] = 3e38
y = np.loadtxt('meta_data.txt', dtype=str, delimiter=',', usecols=[39])

model = RandomForestClassifier()
model.fit(X, y)
with open('meta_learner.pkl', 'wb') as f:
    pickle.dump(model, f)