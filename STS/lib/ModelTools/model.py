import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

class Models:
    def __init__(self):
        pass
    def lr(self):
        return LinearRegression()
    
    def logisticRegression(self):
        return LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')