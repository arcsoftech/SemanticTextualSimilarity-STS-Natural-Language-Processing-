import os
from matplotlib import pyplot as plt
from lib.ModelTools.features import Features
from lib.ModelTools.model import Models
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import scipy.stats as stats

def training(featureObject):
    featureObject.plot(x='word_vectors', y='label', style='o')
    # plt.title('cosine vs label')
    # plt.xlabel('cosine')
    # plt.ylabel('label')
    # plt.show()
    X = featureObject.iloc[:,:-1]  
    Y = featureObject["label"]
    X = scaler.fit_transform(X)
  
    m = Models()
    logistic = m.logisticRegression()
    rf = m.randomForest()
    svm = m.svm()
    gb = m.GB()
    logistic.fit(X, Y)
    rf.fit(X,Y)
    svm.fit(X,Y)
    gb.fit(X,Y)
   
    return logistic,rf,svm,gb


def testing(model, featureObject):
    X = featureObject.iloc[:,:-1]  
    Y = featureObject["label"]
    X = scaler.transform(X)
    Y_pred = model.predict(X)
    print(model.score(X,Y))
    df = pd.DataFrame({'Actual': Y, 'Predicted': Y_pred})
    print(stats.pearsonr(Y_pred,Y))
    return df


if __name__ == "__main__":   
    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, 'Features/')
    dev_feature_set = pd.read_pickle("{}dev".format(directory))
    train_feature_set = pd.read_pickle("{}train".format(directory))
    test_feature_set = pd.read_pickle("{}test".format(directory))
    train_feature_set.drop(['jaccard'], axis=1, inplace=True)
    dev_feature_set.drop(['jaccard'], axis=1, inplace=True)
    print(train_feature_set.loc[0])
    logistic,rf,svm,gb = training(train_feature_set)
    logistic_df=testing(logistic, dev_feature_set)
    svm_df=testing(svm,dev_feature_set)
    rf_df=testing(rf,dev_feature_set)
    gb_df=testing(gb,dev_feature_set)

    # logistic_df.to_csv("result_logistic.csv")
    # rf_df.to_csv("result_rf.csv")
    # svm_df.to_csv("result_rf.csv")
   