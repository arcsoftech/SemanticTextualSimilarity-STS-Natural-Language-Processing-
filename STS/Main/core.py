
from features import cosine_simlarity
from matplotlib import pyplot as plt
from model import Models
from sklearn import metrics
import pandas as pd
import numpy as np
import os

def training(devset):
    featureObject = Preprocessing(devset)
    devset['cosineScore'] = devset.apply(lambda row: cosine_simlarity(
        featureObject.sent_vector(row.Sentence1), featureObject.sent_vector(row.Sentence2)), axis=1)
    devset.plot(x='cosineScore', y='Gold Tag', style='o')
    # plt.title('cosineScore vs Gold Tag')
    # plt.xlabel('cosineScore')
    # plt.ylabel('Gold Tag')
    # plt.show()
    X = devset['cosineScore'].values.reshape(-1, 1)
    Y = devset['Gold Tag'].values.reshape(-1, 1)
    m = Models()
    lr = m.lr()
    lr.fit(X, Y)
    return lr


def testing(model, testset):
    featureObject = Preprocessing(testset)
    testset['cosineScore'] = testset.apply(lambda row: cosine_simlarity(
        featureObject.sent_vector(row.Sentence1), featureObject.sent_vector(row.Sentence2)), axis=1)
    X = testset['cosineScore'].values.reshape(-1, 1)
    Y = testset['Gold Tag'].values.reshape(-1, 1)
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
    # lr = training(train_set)
    # testing(lr, dev_set)
