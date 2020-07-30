import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from util import dsloader

# load the dataset
df_training = pd.read_csv(filepath_or_buffer='../data/training.csv', sep=';', header=0)
df_test = pd.read_csv(filepath_or_buffer='../data/test.csv', sep=';', header=0)
df_full = df_training.append(df_test, ignore_index=True)

# standardize the numeric features
numeric_features_indeces = [0, 10, 11, 12, 13, 15, 16, 17, 18, 19]
for index in numeric_features_indeces:
    df_full.iloc[:, index] = scale(df_full.iloc[:, index])

# replace the categorical features with dummy variables
df_Y = df_full.iloc[:, 20]
df_features = df_full.drop(df_full.columns[20], axis=1)
df_features_dummies = pd.get_dummies(df_features)
df_features_dummies[len(df_features_dummies.columns)] = df_Y

# perform the PCA
df = df_features_dummies
df_Y = df.iloc[:, len(df.columns) - 1]
df.drop(
    df.columns[len(df.columns) - 1], axis=1, inplace=True)
pca = PCA(n_components=27)
df_pca = pd.DataFrame(
    pca.fit_transform(df))
df_pca[len(df_pca.columns)] = df_Y

COMP_X = 1
COMP_Y = 2
fig = plt.figure(figsize=(8,4))

# plot the components
data_chart = fig.add_subplot(1,2,1)
data_chart.set_xlabel("Comp {} - {:.02f}%".format(COMP_X + 1, pca.explained_variance_ratio_[COMP_X] * 100), fontsize=12)
data_chart.set_ylabel("Comp {} - {:.02f}%".format(COMP_Y + 1, pca.explained_variance_ratio_[COMP_Y] * 100), fontsize=12)
data_chart.set_title("PCA Components ({}, {})".format(COMP_X + 1, COMP_Y + 1), fontsize=16)
targets = ["yes", "no"]
colors = ["r", "k"]
for target, color in zip(targets,colors):
    indicesToKeep = df_pca.iloc[:, len(df_pca.columns)-1] == target
    data_chart.scatter(df_pca.loc[indicesToKeep, COMP_X],
                df_pca.loc[indicesToKeep, COMP_Y],
                c = color,
                s = 20)
data_chart.legend(targets)

# plot the feature loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_) # https://stackoverflow.com/a/44728692 https://stats.stackexchange.com/a/143949
loadings_df = pd.DataFrame(loadings)
loadings_chart = fig.add_subplot(1,2,2)
loadings_chart.set_xlabel("Comp {}".format(COMP_X + 1), fontsize=12)
loadings_chart.set_ylabel("Comp {}".format(COMP_Y + 1), fontsize=12)
loadings_chart.set_title("Feature Loadings", fontsize=16)
loadings_chart.scatter(loadings_df.iloc[:, COMP_X],
            loadings_df.iloc[:, COMP_Y],
            c = "k",
            s = 10)
for i in range(loadings_df.shape[0]):
    loadings_chart.annotate(df.columns[i], (loadings_df.iloc[i, COMP_X], loadings_df.iloc[i, COMP_Y]))

plt.show()