import pandas as pd

class CorpusReader:
    def  __init__(self, path):
        self.dev  = pd.read_csv(path+"/dev-set.txt", sep='\t',quoting=3)
        self.train = pd.read_csv(path+"/train-set.txt", sep='\t',quoting=3)
        self.test = pd.read_csv(path+"/test-set.txt", sep='\t',quoting=3) 
        self.data= pd.read_csv(path+"/text.txr", sep="\t",quoting=3)

    def get (self,dev=0):
        if dev == 0:
            return self.data
        elif dev == 1:
            return self.train
        else:
            return self.test

