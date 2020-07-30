import pandas as pd
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

RESPONSE_COLUMN_INDEX = 27
RESPONSE_COLUMN_INDEX_NO_PCA = 63

def load_training_test():
    ###Loads the training and test datasets in a pd DataFrame###
    df_training = pd.read_csv(filepath_or_buffer='../data/training.csv', sep=';', header=0)
    df_test = pd.read_csv(filepath_or_buffer='../data/test.csv', sep=';', header=0)
    
    # standardize the numeric features
    numeric_features_indeces = [0, 10, 11, 12, 13, 15, 16, 17, 18, 19]
    for index in numeric_features_indeces:
        df_training.iloc[:, index] = scale(df_training.iloc[:, index])
        df_test.iloc[:, index] = scale(df_test.iloc[:, index])

    df_full = df_training.append(df_test, ignore_index=True)

    # replace the categorical features with dummy variables
    df_Y = df_full.iloc[:, 20]
    df_features = df_full.drop(df_full.columns[20], axis=1)
    df_features_dummies = pd.get_dummies(df_features)
    
    # perform PCA
    pca = PCA(n_components=27)
    pca.fit(
        df_features_dummies[:df_training.shape[0]]) # fit to training data only
    df_full = pd.DataFrame(
        pca.transform(df_features_dummies)) # transform full dataset
    df_full[len(df_full.columns)] = df_Y

    # split to original training and test
    training_df = df_full[:df_training.shape[0]].reset_index(drop=True)
    test_df = df_full[df_training.shape[0]:].reset_index(drop=True)

    return (training_df, test_df)

def load_training_reduced():
    ###Loads the training_reduced dataset in a pd DataFrame###
    df = pd.read_csv(filepath_or_buffer='../data/training_reduced.csv', sep=';', header=0)

    # standardize the numeric features
    numeric_features_indeces = [0, 10, 11, 12, 13, 15, 16, 17, 18, 19]
    for index in numeric_features_indeces:
        df.iloc[:, index] = scale(df.iloc[:, index])

    # replace the categorical features with dummy variables
    df_Y = df.iloc[:, 20]
    df_features = df.drop(df.columns[20], axis=1)
    df_features_dummies = pd.get_dummies(df_features)
    
    # perform PCA
    pca = PCA(n_components=27)
    pca.fit(df_features_dummies)
    df = pd.DataFrame(
        pca.transform(df_features_dummies))
    df[len(df.columns)] = df_Y

    return df

def load_training_test_no_pca():
    ###Loads the training and test datasets in a pd DataFrame, without doing PCA transformation###
    df_training = pd.read_csv(filepath_or_buffer='../data/training.csv', sep=';', header=0)
    df_test = pd.read_csv(filepath_or_buffer='../data/test.csv', sep=';', header=0)

    # standardize the numeric features
    numeric_features_indeces = [0, 10, 11, 12, 13, 15, 16, 17, 18, 19]
    for index in numeric_features_indeces:
        df_training.iloc[:, index] = scale(df_training.iloc[:, index])
        df_test.iloc[:, index] = scale(df_test.iloc[:, index])

    df_full = df_training.append(df_test, ignore_index=True)

    # replace the categorical features with dummy variables
    df_Y = df_full.iloc[:, 20]
    df_features = df_full.drop(df_full.columns[20], axis=1)
    df_features_dummies = pd.get_dummies(df_features)
    df_features_dummies["y"] = df_Y

    # split to original training and test
    training_df = df_features_dummies[:df_training.shape[0]].reset_index(drop=True)
    test_df = df_features_dummies[df_training.shape[0]:].reset_index(drop=True)
    
    return (training_df, test_df)

def load_training_reduced_no_pca():
    ###Loads the training_reduced dataset in a pd DataFrame, without doing PCA transformation###
    df = pd.read_csv(filepath_or_buffer='../data/training_reduced.csv', sep=';', header=0)

    # standardize the numeric features
    numeric_features_indeces = [0, 10, 11, 12, 13, 15, 16, 17, 18, 19]
    for index in numeric_features_indeces:
        df.iloc[:, index] = scale(df.iloc[:, index])

    # replace the categorical features with dummy variables
    df_Y = df.iloc[:, 20]
    df_features = df.drop(df.columns[20], axis=1)
    df_features_dummies = pd.get_dummies(df_features)
    df_features_dummies["y"] = df_Y

    return df_features_dummies
