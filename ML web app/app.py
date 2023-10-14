import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
model2 = pickle.load(open("model2.pkl", "rb"))
@flask_app.route('/')
def home():
    return render_template('home.html')

@flask_app.route('/serum_data')
def serum_data():
    return render_template('index.html')
@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "Your kidney status is 99% {}".format(prediction))
@flask_app.route('/urine_data')
def urine_data():
    return render_template('Urine.html')
@flask_app.route("/Upredict", methods = ["POST"])
def Upredict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model2.predict(features)
    return render_template("Urine.html", prediction_text = "Your kidney status is 98% {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)