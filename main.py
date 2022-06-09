# Import libraries
import numpy as np
from flask import Flask, request, jsonify,render_template
import pickle
import sklearn
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl','rb'))
@app.route("/predict",methods=['POST'])
def predict():
    
    json= request.json
    query_df = pd.Dataframe(json)
    prediction = model.predict(query_df)
    return jsonify({"prediction":list(prediction)})


if __name__ == '__main__':
    app.run(port=5000, debug=True)