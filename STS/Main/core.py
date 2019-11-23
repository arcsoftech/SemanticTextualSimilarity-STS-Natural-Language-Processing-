
from features import Features
from matplotlib import pyplot as plt
from model import Models
from sklearn import metrics
import pandas as pd
import numpy as np
import os

def training(devset):
    featureObject = Features(devset).generate()
    print(featureObject)
    featureObject.plot(x='cosine', y='label', style='o')
    plt.title('cosine vs label')
    plt.xlabel('cosine')
    plt.ylabel('label')
    plt.show()
    X = featureObject['cosine'].values.reshape(-1, 1)
    Y = featureObject['label'].values.reshape(-1, 1)
    m = Models()
    lr = m.lr()
    lr.fit(X, Y)
    return lr


def testing(model, testset):
    featureObject = Features(testset).generate()
 
    X = featureObject['cosine'].values.reshape(-1, 1)
    Y = featureObject['label'].values.reshape(-1, 1)
    Y_pred = model.predict(X)
    df = pd.DataFrame({'Actual': Y.flatten(), 'Predicted': Y_pred.flatten()})
    print(df)
    print('Mean Squared Error:', metrics.mean_squared_error(Y, Y_pred))
    print(lr.score(X, Y))


if __name__ == "__main__":
    
    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, '../PreProcessesData/')
    dev_set = pd.read_pickle("{}dev".format(directory))
    train_set = pd.read_pickle("{}train".format(directory))
    test_set = pd.read_pickle("{}test".format(directory))
    print(dev_set.loc[0])
    lr = training(train_set)
    testing(lr, dev_set)
