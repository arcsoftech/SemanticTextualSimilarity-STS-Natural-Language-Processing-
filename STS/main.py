import os
from matplotlib import pyplot as plt
from lib.ModelTools.features import Features
from lib.ModelTools.model import Models
from sklearn import metrics
import pandas as pd
import numpy as np

def training(featureObject):
    # featureObject.plot(x='cosine', y='label', style='o')
    # plt.title('cosine vs label')
    # plt.xlabel('cosine')
    # plt.ylabel('label')
    # plt.show()
    X = featureObject.iloc[:,:-1]  
    Y = featureObject["label"]
    print(X)
    print(Y)
    m = Models()
    logistic = m.logisticRegression()
    gmm = m.gaussianMixture()
    logistic.fit(X, Y)
    gmm.fit(X,Y)
    return logistic,gmm


def testing(model, featureObject):
    X = featureObject.iloc[:,:-1]  
    Y = featureObject["label"]
    Y_pred = model.predict(X)
    df = pd.DataFrame({'Actual': Y, 'Predicted': Y_pred})
    return df


if __name__ == "__main__":   
    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, 'Features/')
    dev_feature_set = pd.read_pickle("{}dev".format(directory))
    train_feature_set = pd.read_pickle("{}train".format(directory))
    test_feature_set = pd.read_pickle("{}test".format(directory))
    logistic,gmm = training(train_feature_set)
    logistic_df=testing(logistic, dev_feature_set)
    gmm_df=testing(gmm,dev_feature_set)
    logistic_df.to_csv("result_logistic.csv")
    gmm_df.to_csv("result_gmm.csv")
   