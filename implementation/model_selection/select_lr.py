import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from util import dsloader
from util import utils
frame = dsloader.load_training_reduced()
transform_y_to_numeric = lambda x: 1 if x == "yes" else 0
transform_numeric_to_y = lambda x: "yes" if x >= 0.5 else "no"
frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX] = frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].apply(transform_y_to_numeric) # encode response from 0 to 1 so that LR could be applied, then use 0.5 as discrimination threshold
label_encoder = LabelEncoder()

K = 10  # k-fold cross validation
c_candidates = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
results_df = pd.DataFrame(np.zeros((2, len(c_candidates))))
i = 0
for c_candidate in c_candidates:
    random_split = utils.cv_split(frame, K)
    current_auc = 0
    for j in range(len(random_split)):
        test = random_split[j]
        training_list = random_split[0:j] + random_split[j+1:len(random_split)]
        training = pd.concat(training_list)

        X_train = training.drop(training.columns[dsloader.RESPONSE_COLUMN_INDEX], axis=1)
        Y_train = training.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].values
        X_test = test.drop(test.columns[dsloader.RESPONSE_COLUMN_INDEX], axis=1)
        Y_test = label_encoder.fit_transform(
            test.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].apply(transform_numeric_to_y))
        model = LogisticRegression(C=c_candidate)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        predictions = label_encoder.fit_transform(
            np.array([transform_numeric_to_y(prediction) for prediction in predictions]))
        
        current_auc += roc_auc_score(Y_test, predictions)

    current_auc /= len(random_split)
    print("step {}: {} - {}".format(i + 1, c_candidate, current_auc))
    results_df.iloc[0, i] = c_candidate
    results_df.iloc[1, i] = current_auc
    i += 1

results_df.to_csv("select_lr.csv")