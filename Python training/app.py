import numpy as np
from flask import Flask, request, jsonify, render_template
import re
import pickle

flask_app = Flask(__name__)
model = pickle.load(open("trained_model.pkl","rb"))
cv = pickle.load(open('feature_extr.pkl','rb'))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods= ["POST"])
def predict():
    float_features = cv.transform(request.form.values())
    prediction =  model.predict(float_features)
    return render_template("index.html", prediction_text = format(prediction) )

if __name__ == "__main__":
    flask_app.run(debug=True)