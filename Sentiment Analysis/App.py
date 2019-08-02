from flask import Flask, render_template,request, redirect, url_for, jsonify
from flask_restful import reqparse, abort, Api, Resource
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from Prediction import Prediction
import os
current_path = os.getcwd()
app = Flask(__name__)
api = Api(app)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=["POST"])
def predict():

    # data = request.get_json(force = True)
    # predictor = Prediction()
    # model, vectorizer = predictor.prediction()
    # result = predictor.predict(data['input'], model, vectorizer)
    # return jsonify(result)

   text = request.form["input"]
   predictor = Prediction()
   model, vectorizer = predictor.prediction()
   result = predictor.predict(text, model, vectorizer)
   if result == 1:
       str = "Positive Review"
   else:
       str = "Negative Review"
   return render_template("result.html",result = str)


if __name__ == '__main__':
	app.run(port= 6003,debug=True)
