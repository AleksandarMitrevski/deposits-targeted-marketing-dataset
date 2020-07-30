import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from util import dsloader
from util import utils
training_frame, test_frame = dsloader.load_training_test()
label_encoder = LabelEncoder()
transform_y_to_numeric = lambda x: 1 if x == "yes" else 0
transform_numeric_to_y = lambda x: "yes" if x >= 0.5 else "no"
training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX] = training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].apply(transform_y_to_numeric) # encode response from 0 to 1 so that LR could be applied, then use 0.5 as discrimination threshold
test_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX] = test_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].apply(transform_y_to_numeric)
output = open("evaluate_lr.txt", "w")

SELECTED_HYPERPARAM_VALUE = 5

# evaluation on training dataset
print("Evaluating on training...")
X_train = training_frame.drop(training_frame.columns[dsloader.RESPONSE_COLUMN_INDEX], axis=1)
Y_train = training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].values
X_test = X_train
Y_test = label_encoder.fit_transform(
            training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].apply(transform_numeric_to_y))
model = LogisticRegression(C=SELECTED_HYPERPARAM_VALUE)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
predictions = label_encoder.fit_transform(
            np.array([transform_numeric_to_y(prediction) for prediction in predictions]))

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
Y_test = label_encoder.fit_transform(
            test_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].apply(transform_numeric_to_y))
model = LogisticRegression(C=SELECTED_HYPERPARAM_VALUE)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
predictions = label_encoder.fit_transform(
            np.array([transform_numeric_to_y(prediction) for prediction in predictions]))

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
