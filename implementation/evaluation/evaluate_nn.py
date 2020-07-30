import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as tf_backend
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from util import dsloader
from util import utils
training_frame, test_frame = dsloader.load_training_test()
label_encoder = LabelEncoder()
transform_y_to_numeric = lambda x: 1 if x == "yes" else 0
transform_numeric_to_y = lambda x: "yes" if x >= 0.5 else "no"
training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX] = training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].apply(transform_y_to_numeric) # encode response from 0 to 1 so that the NN could be applied
test_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX] = test_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].apply(transform_y_to_numeric) 
output = open("evaluate_nn.txt", "w")

SELECTED_EPOCHS = 100
SELECTED_NODES_PER_LAYER = 18   # two thirds of number of PCA components
SELECTED_HIDDEN_LAYERS = 1
SELECTED_NN_COUNT = 7

# evaluation on training dataset
print("Evaluating on training...")
X_train = training_frame.drop(training_frame.columns[dsloader.RESPONSE_COLUMN_INDEX], axis=1)
Y_train = training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].values
X_test = X_train
Y_test = label_encoder.fit_transform(
            training_frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].apply(transform_numeric_to_y))
TP = 0
FP = 0
TN = 0
FN = 0
AUC = 0
for i in range(SELECTED_NN_COUNT):
    tf_backend.clear_session() # clear keras session
    print("\n\t Training NN #{}".format(i + 1))
    model = Sequential()
    for _ in range(SELECTED_HIDDEN_LAYERS):
        model.add(Dense(SELECTED_NODES_PER_LAYER, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam")

    model.fit(x=X_train, y=Y_train, epochs=SELECTED_EPOCHS, verbose=0)
    predictions = model.predict(X_test)
    predictions = label_encoder.fit_transform(
        np.array([transform_numeric_to_y(prediction) for prediction in predictions]))

    current_confusion_matrix = confusion_matrix(Y_test, predictions)
    TP += current_confusion_matrix[1,1]
    FP += current_confusion_matrix[0,1]
    TN += current_confusion_matrix[0,0]
    FN += current_confusion_matrix[1,0]
    AUC += roc_auc_score(Y_test, predictions)

TP = int(round(TP / SELECTED_NN_COUNT))
FP = int(round(FP / SELECTED_NN_COUNT))
TN = int(round(TN / SELECTED_NN_COUNT))
FN = int(round(FN / SELECTED_NN_COUNT))
AUC /= SELECTED_NN_COUNT

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
TP = 0
FP = 0
TN = 0
FN = 0
AUC = 0
for i in range(SELECTED_NN_COUNT):
    tf_backend.clear_session()
    print("\n\t Training NN #{}".format(i + 1))
    model = Sequential()
    for _ in range(SELECTED_HIDDEN_LAYERS):
        model.add(Dense(SELECTED_NODES_PER_LAYER, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam")

    model.fit(x=X_train, y=Y_train, epochs=SELECTED_EPOCHS, verbose=0)
    predictions = model.predict(X_test)
    predictions = label_encoder.fit_transform(
        np.array([transform_numeric_to_y(prediction) for prediction in predictions]))

    current_confusion_matrix = confusion_matrix(Y_test, predictions)
    TP += current_confusion_matrix[1,1]
    FP += current_confusion_matrix[0,1]
    TN += current_confusion_matrix[0,0]
    FN += current_confusion_matrix[1,0]
    AUC += roc_auc_score(Y_test, predictions)

TP = int(round(TP / SELECTED_NN_COUNT))
FP = int(round(FP / SELECTED_NN_COUNT))
TN = int(round(TN / SELECTED_NN_COUNT))
FN = int(round(FN / SELECTED_NN_COUNT))
AUC /= SELECTED_NN_COUNT

line = "***Test results***\nTP = {}; FP = {}; TN = {}; FN = {}; AUC = {}".format(TP, FP, TN, FN, AUC)
output.write("\n" + line + "\n")
print("\n" + line)

output.close()
