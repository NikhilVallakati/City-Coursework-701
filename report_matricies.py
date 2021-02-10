import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

class Report_Matricies:

    @staticmethod
    def accuracy(target_test, final_accuracy):
        conmat = np.array(confusion_matrix(target_test, final_accuracy, labels=[1,0]))
        confusion = pd.DataFrame(conmat, index=['positive', 'negative'],
                                columns=['predicted_positive','predicted_negative'])
        print("-"*80)
        print("Confusion Matrix\n")
        print(confusion)
        print("-"*80)
        print("Classification Report\n")
        print(classification_report(target_test, final_accuracy))