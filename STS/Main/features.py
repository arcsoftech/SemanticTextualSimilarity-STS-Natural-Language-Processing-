import pandas as pd
def bag_of_words(vocabulary,sentence):

    #construct bag of words
    l1 = []
    for w in vocabulary:
        if w in sentence:
            l1.append(1)
        else:
            l1.append(0)
    return l1

class Features:
    def __init__(self,df):
        self.df = df
        self.data = pd.DataFrame()
    
    def generate(self):
        self.data["cosine"] = self.df.apply(self.__cosine_simlarity__,axis=1)
        self.data["label"] = self.df.apply(self.__get_gold_tag__,axis=1)
        return self.data
        
    
    def __get_gold_tag__(self,row):
        return row['Gold Tag']

    def __cosine_simlarity__(self,row):
        # cosine formula
        v1 = bag_of_words(row['vocabulary'],row["Sentence1"])
        v2 = bag_of_words(row['vocabulary'],row["Sentence2"])
        c = 0
        for i in range(len(v1)):
            c += v1[i]*v2[i]
        cosine = c / float((sum(v1)*sum(v2))**0.5)
        return cosine
