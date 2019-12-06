import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture as GMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import  GradientBoostingClassifier
class Models:
    def __init__(self):
        pass
    def randomForest(self):
        classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
        return classifier
    def svm(self):
        svm_model_linear = SVC(kernel = 'rbf', C = 1)
        return svm_model_linear
    def GB(self):
        gb = GradientBoostingClassifier(n_estimators=100,criterion='friedman_mse')
        return gb

    