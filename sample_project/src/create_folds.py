import numpy as np
import pandas as pd 
from sklearn import model_selection
import config


def create_folds(data):
    data["kfold"] = -1

    #shuffling data
    data = data.sample(frac = 1).reset_index(drop = True)

    #intializing stratified kfold
    kf = model_selection.KFold(n_splits = 5)

    for f,(t_,v_) in enumerate(kf.split(X = data)):
        data.loc[v_,"kfold"] = f

    # save the csv with kfold column
    data.to_csv(config.TRAINING_FOLDS_FILE, index = False)

if __name__ == "__main__":
    df_train = pd.read_csv(config.TRAINING_FILE)
    create_folds(df_train)
