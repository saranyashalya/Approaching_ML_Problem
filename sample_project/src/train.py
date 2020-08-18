import joblib
import pandas as pd 
from sklearn import metrics
from sklearn import tree 
import config
import os
import argparse
import model_dispatcher

def run(fold , model):
    df =  pd.read_csv(config.TRAINING_FOLDS_FILE)
    
    #selecting training and validation dataset
    df_train = df[df.kfold != fold].reset_index(drop = True)
    
    df_valid = df[df.kfold == fold].reset_index(drop = True)
    
    #converting numpy arrays from dataframe
    x_train = df_train.drop('label', axis = 1).values
    y_train = df_train.label.values
    
    x_valid = df_valid.drop('label', axis = 1).values
    y_valid = df_valid.label.values
    
    #Fitting model
    clf = model_dispatcher.models[model]
    
    #training model
    clf.fit(x_train, y_train)
    
    #prediction
    pred = clf.predict(x_valid)
    
    #accuracy 
    accuracy = metrics.accuracy_score(y_valid, pred)
    print(f"Fold = {fold}, Accuracy = {accuracy}")
    
    #save the model
    joblib.dump(
                clf, 
                os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )
    
if __name__ == '__main__':
    
    #initialize parser
    parser = argparse.ArgumentParser()
    
    #add the arguments
    parser.add_argument(
        "--fold",
        type = int,
    )
    
    parser.add_argument(
        "--model",
        type = str,
    )
    
    args = parser.parse_args()
    
    run(fold = args.fold,
        model = args.model)
