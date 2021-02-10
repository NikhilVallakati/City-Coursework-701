import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs

from logistic_regression import Logistic_Regression
from local_time import LocalTime
from report_matricies import Report_Matricies

class N_Gram:
    @staticmethod
    def calc(args, no_of_words):
        print("-"*100)
        print(LocalTime.get(), "  Words selected report: NGram where n = ", no_of_words)
        print("-"*100)
        ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, no_of_words))
        X = ngram_vectorizer.fit_transform(args.Train_text)
        X_test = ngram_vectorizer.transform(args.Test_text)
        best_c = Logistic_Regression.get_best_hyperparameter(args.X_train, args.Y_train, args.Y_val, args.X_val)
        final_ngram = LogisticRegression(C=best_c)
        final_ngram.fit(X, args.Target)
        final_accuracy = final_ngram.predict(X_test)
        final_accuracy_score = accuracy_score(args.Target_test, final_accuracy)
        print ("Final NGram Accuracy: %s" % final_accuracy_score)
        Report_Matricies.accuracy(args.Target_test, final_accuracy)
        feature_names = zip(args.Cv.get_feature_names(), final_ngram.coef_[0])
        feature_to_coef = {
            word: coef for word, coef in feature_names
        }
        itemz = feature_to_coef.items()
        list_positive = sorted(
            itemz, 
            key=lambda x: x[1], 
            reverse=True)
        print("-"*100)
        print(LocalTime.get(), "--- Most popular positve words")
        for best_positive in list_positive[:args.Number_we_are_interested_in]:
            print (best_positive)
        print("-"*100)
        print(LocalTime.get(), "--- Most popular negative words")
        list_negative = sorted(
            itemz, 
            key=lambda x: x[1])
        for best_negative in list_negative[:args.Number_we_are_interested_in]:
            print (best_negative)