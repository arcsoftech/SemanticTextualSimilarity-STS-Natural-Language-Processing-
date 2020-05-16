import os
import sys
from lib.ModelTools.features import Features
from lib.ModelTools.model import Models
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import scipy.stats as stats
from sklearn.metrics import precision_recall_fscore_support

from joblib import dump, load
m = Models()

def training(featureObject,model,modelName):
    featureObject.plot(x='word_vectors', y='label', style='o')
    X = featureObject.iloc[:,:-1]  
    Y = featureObject["label"]
    Scalermodel=scaler.fit(X)
    X= Scalermodel.transform(X)

    dirname = os.path.dirname(__file__)
    storepath = os.path.join(dirname,"Models")
    try:
        os.stat(storepath)
    except:
        os.mkdir(storepath)

    model.fit(X,Y)
    dump(model,"{}/{}".format(storepath,modelName))
    return model


def testing(model, modelName,featureObject):
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
    id=["s_{}".format(k+1) for k in range(len(Y_pred))]
    df = pd.DataFrame({'id': id, 'Gold Tag': Y_pred})
    df.to_csv("{}/{}.txt".format(storepath,modelName),sep="\t",index=False)
    return Y_pred


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
    if(op in [0,1,2]):
        svm = training(train_feature_set,m.svm(),"svm")
        rf = training(train_feature_set,m.randomForest(),"rf")
        gb = training(train_feature_set,m.GB(),"gb")
    # if(op == 2):
    #     loadpath = os.path.join(dirname, 'Models')
    #     rf = load("{}/rf".format(loadpath))
    #     svm = load("{}/svm".format(loadpath))
    #     gb = load("{}/gb".format(loadpath))
    if (op == 1):
        svm_df=testing(svm,"svm_dev",dev_feature_set)
        rf_df=testing(rf,"rf_dev",dev_feature_set)
        gb_df=testing(gb,"gb_dev",dev_feature_set)
        final = np.rint(np.mean(np.array([svm_df,rf_df,gb_df]), axis=0))
        id=["s_{}".format(k+1) for k in range(len(final))]
        df = pd.DataFrame({'id': id, 'Gold Tag': [int(x) for x in final]})
        df.to_csv("{}/{}.txt".format(os.path.join(dirname, 'Predictions'),"final_dev"),sep="\t",index=False)
        print("{}\n{}".format("Final",stats.pearsonr(final,dev_feature_set["label"])))
        p,r,f,s=precision_recall_fscore_support(dev_feature_set["label"],final, average='macro')
        print("Precission:{}\nRecall:{}\nFScore:{}\nSupport:{}".format(p,r,f,s))

    if (op == 2):
        svm_df=testing(svm,"svm_test",test_feature_set)
        rf_df=testing(rf,"rf_test",test_feature_set)
        gb_df=testing(gb,"gb_test",test_feature_set)
        final = np.rint(np.mean(np.array([svm_df,rf_df,gb_df]), axis=0))
        id=["p_{}".format(k+1) for k in range(len(final))]
        df = pd.DataFrame({'id': id, 'Gold Tag': [int(x) for x in final]})
        df.to_csv("{}/{}.txt".format(os.path.join(dirname, 'Predictions'),"final_test"),sep="\t",index=False)
   