import numpy as np

def cv_split(df, K):
    ###Returns training and test DataFrame splits for k-fold cross validation###

    # copy the dataframe with randomized rows
    clone_df = df.sample(frac=1).reset_index(drop=True)

    # split and return
    return np.array_split(clone_df, K)
