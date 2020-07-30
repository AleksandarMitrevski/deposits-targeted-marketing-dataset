import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from util import dsloader
from util import utils
training_frame, test_frame = dsloader.load_training_test_no_pca()
label_encoder = LabelEncoder()
transform_y_to_numeric = lambda x: 1 if x == "yes" else -1
transform_numeric_to_y = lambda x: "yes" if x >= 0 else "no"
training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX_NO_PCA] = training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX_NO_PCA].apply(transform_y_to_numeric) # encode response from -1 to 1 so that PLS regression could be applied, then use 0 as discrimination threshold
test_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX_NO_PCA] = test_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX_NO_PCA].apply(transform_y_to_numeric)
output = open("evaluate_plsda.txt", "w")

SELECTED_NUMBER_OF_FEATURES = 6

# evaluation on training dataset
print("Evaluating on training...")
X_train = training_frame.drop(training_frame.columns[dsloader.RESPONSE_COLUMN_INDEX_NO_PCA], axis=1)
Y_train = training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX_NO_PCA].values
X_test = X_train
Y_test = label_encoder.fit_transform(
            training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX_NO_PCA].apply(transform_numeric_to_y))
model = PLSRegression(n_components=SELECTED_NUMBER_OF_FEATURES)
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
X_train = training_frame.drop(training_frame.columns[dsloader.RESPONSE_COLUMN_INDEX_NO_PCA], axis=1)
Y_train = training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX_NO_PCA].values
X_test = test_frame.drop(test_frame.columns[dsloader.RESPONSE_COLUMN_INDEX_NO_PCA], axis=1)
Y_test = label_encoder.fit_transform(
            test_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX_NO_PCA].apply(transform_numeric_to_y))
model = PLSRegression(n_components=SELECTED_NUMBER_OF_FEATURES)
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
