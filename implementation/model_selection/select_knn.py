import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from util import dsloader
from util import utils
frame = dsloader.load_training_reduced()
label_encoder = LabelEncoder()
frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX] = label_encoder.fit_transform(frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX])

K = 10  # k-fold cross validation
k_candidates = [1, 3, 5, 7, 9, 11, 13, 15, 17]
results_df = pd.DataFrame(np.zeros((2, len(k_candidates))))
i = 0
for k_candidate in k_candidates:
    random_split = utils.cv_split(frame, K)
    current_auc = 0
    for j in range(len(random_split)):
        test = random_split[j]
        training_list = random_split[0:j] + random_split[j+1:len(random_split)]
        training = pd.concat(training_list)

        X_train = training.drop(training.columns[dsloader.RESPONSE_COLUMN_INDEX], axis=1)
        Y_train = training.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].values
        X_test = test.drop(test.columns[dsloader.RESPONSE_COLUMN_INDEX], axis=1)
        Y_test = test.iloc[:, dsloader.RESPONSE_COLUMN_INDEX].values
        classifier = KNeighborsClassifier(n_neighbors=k_candidate)
        classifier.fit(X_train, Y_train)
        predictions = classifier.predict(X_test)
        
        current_auc += roc_auc_score(Y_test, predictions)

    current_auc /= len(random_split)
    print("step {}: {} - {}".format(i + 1, k_candidate, current_auc))
    results_df.iloc[0, i] = k_candidate
    results_df.iloc[1, i] = current_auc
    i += 1

results_df.to_csv("select_knn.csv")
