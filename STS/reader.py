import pandas as pd

class Reader:
    def  __init__(self, path):
        self.dev  = pd.read_csv(path+"/dev-set.txt", sep='\t', lineterminator='\n')
        self.test = pd.read_csv(path+"/train-set.txt", sep='\t', lineterminator='\r')

    def get (self,dev=1)-> None:
        if(dev):
            return self.dev
        else:
            return self.test