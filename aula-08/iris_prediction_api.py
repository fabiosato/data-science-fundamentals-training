from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

clf = pickle.load(open('iris_random_forest_model.pickle', 'rb'))

classes = ['setosa', 'versicolor', 'virginica']

@app.route("/predict")
def predict():
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    data = []
    for f in features:
        data.append(float(request.args.get(f)))

    df = pd.DataFrame(data)
    print(df.T)
    y_pred = clf.predict(df.T)

    return jsonify({'iris': classes[y_pred[0]]})
