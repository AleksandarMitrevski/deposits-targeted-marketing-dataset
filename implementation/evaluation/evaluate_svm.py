import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from util import dsloader
from util import utils
training_frame, test_frame = dsloader.load_training_test()
label_encoder = LabelEncoder()
training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX] = label_encoder.fit_transform(training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX])  # labels are sorted lexically, therefor "no" is always 0 and "yes" is always 1
test_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX] = label_encoder.fit_transform(test_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX])
output = open("evaluate_svm.txt", "w")

SELECTED_HYPERPARAM_VALUE = 1

# evaluation on training dataset
print("Evaluating on training...")
X_train = training_frame.drop(training_frame.columns[dsloader.RESPONSE_COLUMN_INDEX], axis=1)
Y_train = training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].values
X_test = X_train
Y_test = Y_train
classifier = SVC(C=SELECTED_HYPERPARAM_VALUE)
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_test)

current_confusion_matrix = confusion_matrix(Y_test, predictions)
TP = current_confusion_matrix[1,1]
FP = current_confusion_matrix[0,1]
TN = current_confusion_matrix[0,0]
FN = current_confusion_matrix[1,0]
AUC = roc_auc_score(Y_test, predictions)
line = "***Training results***\nTP = {}; FP = {}; TN = {}; FN = {}; AUC = {}".format(TP, FP, TN, FN, AUC)
output.write(line + "\n")
print("\n" + line)

# evaluation on test dataset
print("\nEvaluating on test...")
X_train = training_frame.drop(training_frame.columns[dsloader.RESPONSE_COLUMN_INDEX], axis=1)
Y_train = training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].values
X_test = test_frame.drop(test_frame.columns[dsloader.RESPONSE_COLUMN_INDEX], axis=1)
Y_test = test_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].values
classifier = SVC(C=SELECTED_HYPERPARAM_VALUE)
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_test)

current_confusion_matrix = confusion_matrix(Y_test, predictions)
TP = current_confusion_matrix[1,1]
FP = current_confusion_matrix[0,1]
TN = current_confusion_matrix[0,0]
FN = current_confusion_matrix[1,0]
AUC = roc_auc_score(Y_test, predictions)
line = "***Test results***\nTP = {}; FP = {}; TN = {}; FN = {}; AUC = {}".format(TP, FP, TN, FN, AUC)
output.write("\n" + line + "\n")
print("\n" + line)

output.close()
