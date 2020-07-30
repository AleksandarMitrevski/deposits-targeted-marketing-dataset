import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as tf_backend
import matplotlib.pyplot as plt
import shap

# load training reduced (the procedure downsamples even from this)
df_full = pd.read_csv(filepath_or_buffer='../data/training_reduced.csv', sep=';', header=0)

# create split
training_size = int(round(df_full.shape[0] * 0.8))
training_df = df_full[:training_size].reset_index(drop=True)
X_train = training_df.drop(training_df.columns[20], axis=1)
Y_train = training_df.iloc[:, 20]
testing_df = df_full[training_size:].reset_index(drop=True)
X_test = testing_df.drop(testing_df.columns[20], axis=1)
Y_test = testing_df.iloc[:, 20]

# standardize the numeric features separately
numeric_features_indeces = [0, 10, 11, 12, 13, 15, 16, 17, 18, 19]
for index in numeric_features_indeces:
    X_train.iloc[:, index] = scale(X_train.iloc[:, index])
    X_test.iloc[:, index] = scale(X_test.iloc[:, index])

# replace the categorical features with dummy variables
df_X_full = X_train.append(X_test, ignore_index=True)
df_X_full = pd.get_dummies(df_X_full)

# split back
X_train = df_X_full[:training_size].reset_index(drop=True)
X_test = df_X_full[training_size:].reset_index(drop=True)

# perform PCA
pca = PCA(n_components=27)
pca.fit(X_train) # fit to training data only
X_train_pca = pd.DataFrame(
    pca.transform(X_train))

# transform training y
transform_y_to_numeric = lambda x: 1 if x == "yes" else 0
Y_train = Y_train.apply(transform_y_to_numeric) # encode response from 0 to 1 so that the NN could be applied

# train the NN ensemble
SELECTED_EPOCHS = 100
SELECTED_NODES_PER_LAYER = 18   # two thirds of number of PCA components
SELECTED_HIDDEN_LAYERS = 1
SELECTED_NN_COUNT = 7

print("\nTraining NN ensemble")
models = [None] * SELECTED_NN_COUNT
for i in range(SELECTED_NN_COUNT):
    # tf_backend.clear_session() # we need to preserve the models
    print("\n\t Training NN #{}".format(i + 1))
    model = Sequential()
    for _ in range(SELECTED_HIDDEN_LAYERS):
        model.add(Dense(SELECTED_NODES_PER_LAYER, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam")

    model.fit(x=X_train_pca, y=Y_train, epochs=SELECTED_EPOCHS, verbose=0)
    models[i] = model

# define the predict probabilities function over the entire ensemble
POSITIVE_CLASS_PROB_INDEX = 1
NEGATIVE_CLASS_PROB_INDEX = 0
def predict_proba(X):
    X_pca = pd.DataFrame(
        pca.transform(X))   # project X onto the PCA space
    predictions = np.zeros(shape=(X_pca.shape[0],1))
    for model in models:
        prediction = model.predict(X_pca)
        predictions += prediction
    predictions /= len(models)
    class_probabilities = pd.DataFrame(
        np.zeros(shape=(predictions.shape[0], 2)))
    class_probabilities.iloc[:, POSITIVE_CLASS_PROB_INDEX] = predictions
    class_probabilities.iloc[:, NEGATIVE_CLASS_PROB_INDEX] = pd.DataFrame(predictions).apply(lambda x: 1 - x)
    return class_probabilities

# use SHAP KernelExplainer to explain test set predictions
USE_MATPLOTLIB = True # False requires IPython and likely additional setup
TRAIN_SUMMARY_N_SAMPLES = 100
MULTISAMPLE_PLOTS_N_TEST_SAMPLES = 100

# provided for IPython / Jupyter consistency
if not USE_MATPLOTLIB:
    shap.initjs()

X_train_samples = shap.sample(X_train, nsamples=TRAIN_SUMMARY_N_SAMPLES) # summarize the X_train with a total of 100 samples
explainer = shap.KernelExplainer(predict_proba, X_train_samples, link="logit")

# find first true positive and true negative samples
Y_test_proba = predict_proba(X_test)
tp_index = -1
tn_index = -1
for i in range(len(Y_test_proba)):
    if Y_test_proba.iloc[i, POSITIVE_CLASS_PROB_INDEX] >= 0.5 and Y_test.iloc[i] == "yes":
        tp_index = i
    if Y_test_proba.iloc[i, POSITIVE_CLASS_PROB_INDEX] < 0.5 and Y_test.iloc[i] == "no":
        tn_index = i
    if tp_index != -1 and tn_index != -1:
        break

# force_plot for single TP sample, if it exists
if tp_index != -1:
    print("TP sample: _test index = {}".format(tp_index))
    sample = X_test.iloc[tp_index, :]
    shap_values = explainer.shap_values(sample)
    shap.force_plot(explainer.expected_value[POSITIVE_CLASS_PROB_INDEX], shap_values[POSITIVE_CLASS_PROB_INDEX], sample, link="logit", show=False, matplotlib=USE_MATPLOTLIB)
    plt.savefig("shap_force_tp_sample.png")

# force_plot for single TN sample, if it exists
if tn_index != -1:
    print("TN sample: _test index = {}".format(tn_index))
    sample = X_test.iloc[tn_index, :]
    shap_values = explainer.shap_values(sample)
    shap.force_plot(explainer.expected_value[POSITIVE_CLASS_PROB_INDEX], shap_values[POSITIVE_CLASS_PROB_INDEX], sample, link="logit", show=False, matplotlib=USE_MATPLOTLIB)
    plt.savefig("shap_force_tn_sample.png")

samples = shap.sample(X_test, nsamples=MULTISAMPLE_PLOTS_N_TEST_SAMPLES)
shap_values = explainer.shap_values(samples)   # processing a sample takes time in the order of seconds (in the specific environment)

# stacked force plot for multiple samples, for the positive class - not supported when using matplotlib instead of ipython / jupyter
#shap.force_plot(explainer.expected_value[POSITIVE_CLASS_PROB_INDEX], shap_values[POSITIVE_CLASS_PROB_INDEX], samples, link="logit", show=False, matplotlib=USE_MATPLOTLIB) # matplotlib=True for multiple samples is not supported at the time of implementation
#plt.savefig("shap_force_multisample.png")

# summary plot for multiple samples
# if it gets drawn incorrectly, try running it in isolation (without drawing out the force plots)
shap.summary_plot(shap_values, samples, plot_type="bar", show=False)  # always matplotlib
plt.savefig("shap_summary.png")
