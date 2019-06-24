#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import string
from sklearn.externals import joblib
import os
current_path = os.getcwd()


class Sentiment_analysis:

    # A pre-processing step to make the texts cleaner and easier to process and a vectorization step to transform these texts into numerical vectors.
    def data_prep(self):
        df = open("amazon_reviews.csv").read()
        data = df.split("\n")
        labels, texts = [], []
        for i, line in enumerate(data):
            content = line.split()
            labels.append(content[0])
            texts.append(" ".join(content[1:]))
        d = {'Label':labels,'Reviews':texts}
        df1 = pd.DataFrame(d)
        return df1

    def data_preprocess(self):
        df2 = self.data_prep()
        # Removal of stopwords
        stop = stopwords.words('english')
        df2["Reviews"] = df2["Reviews"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        # Converting the reviews into lowercase
        df2["Reviews"] = df2["Reviews"].apply(lambda x: " ".join(x.lower() for x in x.split()))
        # Removing punctuation marks
        df1["Reviews"] = df1["Reviews"].str.replace('[^\w\s]',' ')
        # Rare words removal
        freq = pd.Series(' '.join(df2["Reviews"]).split()).value_counts()[-10:]
        freq = list(freq.index)
        df2["Reviews"] = df2["Reviews"].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
        # Tokenization of text data
        Token_list = []
        for i in range(0, len(df2)):
            Token_list.append(TextBlob(df2["Reviews"][i]).words)
        return Token_list

    def split_data(self):
        df2 = self.data_prep()
        # splitting the dataset into training and validation datasets
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df2["Reviews"], df2["Label"], random_state = 42)
        # label encode the target variable
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)
        count_vect = CountVectorizer(analyzer = 'word', token_pattern = r'\w{1,}')
        count_vect.fit(df2["Reviews"])
        #transform the training and validation data using count vectorizer object
        xtrain_count =  count_vect.transform(train_x)
        xvalid_count =  count_vect.transform(valid_x)
        return xtrain_count, xvalid_count, train_y, valid_y, valid_x, count_vect

    def train_model(self):
        xtrain_count, xvalid_count, train_y, valid_y, valid_x = self.split_data()
        # fit the training dataset on the classifier
        model = naive_bayes.MultinomialNB().fit(xtrain_count, train_y)
        return model
