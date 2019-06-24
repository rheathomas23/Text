#!/usr/bin/env python
# coding: utf-8
from sklearn.externals import joblib
import os
import sys
from Training import Sentiment_analysis
from sklearn import metrics


class Prediction():

    def __init__(self):
        self.current_path = os.getcwd()

    def prediction(self):
        
        sent_obj = Sentiment_analysis()
        xtrain_count, xvalid_count, train_y, valid_y, valid_x, count_vect = sent_obj.split_data()
        current_path = self.current_path
        model = None
        vectorizer = None
        try:
            model = joblib.load("/Users/rhea-8992/PycharmProjects/Text/Sentiment_analysis_naive_bayes_model.pkl")
            vectorizer = count_vect
        except:
            xtrain_count, xvalid_count, train_y, valid_y, valid_x, count_vect = sent_obj.split_data()
            model = sent_obj.train_model()
            predictions = model.predict(xvalid_count)
            model = joblib.load("/Users/rhea-8992/PycharmProjects/Text/Sentiment_analysis_naive_bayes_model.pkl")
            vectorizer = count_vect
        return model,vectorizer

    def predict(self,string,model,vectorizer):
        list_of_strings = []
        list_of_strings.append(string)
        res = model.predict(vectorizer.transform(list_of_strings).toarray())
        return res

#pred = Prediction()
#pred.prediction()
