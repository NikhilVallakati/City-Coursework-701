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
from params import Params
from linear_svm import Linear_SVM

class SVM:
    @staticmethod
    def calc(args, X, X_test, final_model):
        print("-"*100)
        print(LocalTime.get(), "  Words selected report: SVM ")
        print("-"*100)
        best_c = Linear_SVM.get_best_hyperparameter(args.X_train, args.Y_train, args.Y_val, args.X_val)
        final_svm  = LinearSVC(C=best_c)
        final_svm.fit(X, args.Target)
        final_accuracy = final_svm.predict(X_test)
        final_accuracy_score = accuracy_score(args.Target_test, final_accuracy)
        print ("Final SVM Accuracy: %s" % final_accuracy_score)
        Report_Matricies.accuracy(args.Target_test, final_accuracy)
        feature_names = zip(args.Cv.get_feature_names(), final_model.coef_[0])
        feature_to_coef = {
            word: coef for word, coef in feature_names
        }
        itemz = feature_to_coef.items()
        list_positive = sorted(
            itemz, 
            key=lambda x: x[1], 
            reverse=True)[:args.Number_we_are_interested_in]
        print("-"*100)
        print(LocalTime.get(), "--- Most popular positve words")
        for best_positive in list_positive:
            print (best_positive)
        print("-"*100)
        print(LocalTime.get(), "--- Most popular negative words")
        list_negative = sorted(
            itemz, 
            key=lambda x: x[1])[:args.Number_we_are_interested_in]
        for best_negative in list_negative:
            print (best_negative)
