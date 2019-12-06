import os
import sys
from matplotlib import pyplot as plt
from lib.ModelTools.features import Features
from lib.ModelTools.model import Models
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import scipy.stats as stats
from joblib import dump, load
m = Models()

def standardization(X):
    scaled = scaler.fit_transform(X)
    return scaled
def training(featureObject,model,modelName):
    featureObject.plot(x='word_vectors', y='label', style='o')
    X = featureObject.iloc[:,:-1]  
    Y = featureObject["label"]
    X = standardization(X)

    dirname = os.path.dirname(__file__)
    storepath = os.path.join(dirname,"Models")
    try:
        os.stat(storepath)
    except:
        os.mkdir(storepath)

    model.fit(X,Y)
    dump(model,"{}/{}".format(storepath,modelName))
    return model


def testing(model, featureObject):
    dirname = os.path.dirname(__file__)
    storepath = os.path.join(dirname,"Predictions")
    try:
        os.stat(storepath)
    except:
        os.mkdir(storepath)
    X = featureObject.iloc[:,:-1]  
    Y = featureObject["label"]
    X = scaler.transform(X)
    Y_pred = model.predict(X)
    prediction = pd.Series(Y_pred)
    print(model.score(X,Y))
    df = pd.DataFrame({'Actual': Y, 'Predicted': Y_pred})
    print(stats.pearsonr(Y_pred,Y))
    return df


if __name__ == "__main__":
    try:
        op = int(sys.argv[1])
    except:
        op=0
    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, 'Features/')
    dev_feature_set = pd.read_pickle("{}dev".format(directory))
    train_feature_set = pd.read_pickle("{}train".format(directory))
    test_feature_set = pd.read_pickle("{}test".format(directory))
    # print(train_feature_set.head(1))
    if(op in [0,1]):
        svm = training(train_feature_set,m.svm(),"svm")
        rf = training(train_feature_set,m.randomForest(),"rf")
        gb = training(train_feature_set,m.GB(),"gb")
    if(op == 2):
        loadpath = os.path.join(dirname, 'Models')
        rf = load("{}/rf".format(loadpath))
        svm = load("{}/svm".format(loadpath))
        gb = load("{}/gb".format(loadpath))
    if (op in [1,2]):
        svm_df=testing(svm,dev_feature_set)
        rf_df=testing(rf,dev_feature_set)
        gb_df=testing(gb,dev_feature_set)
   