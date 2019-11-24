import os
from matplotlib import pyplot as plt
from lib.ModelTools.features import Features
from lib.ModelTools.model import Models
from sklearn import metrics
import pandas as pd
import numpy as np

def training(devset):
    featureObject = Features(devset).generate()
    print(featureObject)
    # featureObject.plot(x='cosine', y='label', style='o')
    # plt.title('cosine vs label')
    # plt.xlabel('cosine')
    # plt.ylabel('label')
    # plt.show()
    X = featureObject.iloc[:,:-1]  
    Y = featureObject["label"]

    print("here")
    print(X)
    print(Y)
    m = Models()
    lr = m.logisticRegression()
    lr.fit(X, Y)
    return lr


def testing(model, testset):
    featureObject = Features(testset).generate()
    X = featureObject.iloc[:,:-1]  
    Y = featureObject["label"]
    Y_pred = model.predict(X)
    print(Y_pred)
    df = pd.DataFrame({'Actual': Y, 'Predicted': Y_pred})
    print(df)
    #print('Mean Squared Error:', metrics.mean_squared_error(Y, Y_pred))
    print(lr.score(X, Y))


if __name__ == "__main__":   
    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, 'PreProcessesData/')
    dev_set = pd.read_pickle("{}dev".format(directory))
    train_set = pd.read_pickle("{}train".format(directory))
    test_set = pd.read_pickle("{}test".format(directory))
    print(dev_set.loc[0])
    lr = training(train_set)
    testing(lr, dev_set)
