import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder

from util import dsloader
from util import utils
frame = dsloader.load_training_reduced_no_pca()
transform_y_to_numeric = lambda x: 1 if x == "yes" else -1
transform_numeric_to_y = lambda x: "yes" if x >= 0 else "no"
frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX_NO_PCA] = frame.iloc[:, dsloader.RESPONSE_COLUMN_INDEX_NO_PCA].apply(transform_y_to_numeric) # encode response from -1 to 1 so that PLS regression could be applied, then use 0 as discrimination threshold
label_encoder = LabelEncoder()

K = 10  # k-fold cross validation
f_num_candidates = range(1, 30)
results_df = pd.DataFrame(np.zeros((2, len(f_num_candidates))))
i = 0
for f_num_candidate in f_num_candidates:
    random_split = utils.cv_split(frame, K)
    current_auc = 0
    for j in range(len(random_split)):
        test = random_split[j]
        training_list = random_split[0:j] + random_split[j+1:len(random_split)]
        training = pd.concat(training_list)

        X_train = training.drop(training.columns[dsloader.RESPONSE_COLUMN_INDEX_NO_PCA], axis=1)
        Y_train = training.iloc[:, dsloader.RESPONSE_COLUMN_INDEX_NO_PCA].values
        X_test = test.drop(test.columns[dsloader.RESPONSE_COLUMN_INDEX_NO_PCA], axis=1)
        Y_test = label_encoder.fit_transform(
            test.iloc[:, dsloader.RESPONSE_COLUMN_INDEX_NO_PCA].apply(transform_numeric_to_y))
        model = PLSRegression(n_components=f_num_candidate)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        predictions = label_encoder.fit_transform(
            np.array([transform_numeric_to_y(prediction) for prediction in predictions]))
        
        current_auc += roc_auc_score(Y_test, predictions)

    current_auc /= len(random_split)
    print("step {}: {} - {}".format(i + 1, f_num_candidate, current_auc))
    results_df.iloc[0, i] = f_num_candidate
    results_df.iloc[1, i] = current_auc
    i += 1

results_df.to_csv("select_plsda.csv")