import pandas as pd
import spacy
try:
    from Preprocessing.featureGenerator import Preprocessing
except ImportError:
    from ..Preprocessing.featureGenerator import Preprocessing


nlp = spacy.load("en_core_web_lg")
def bag_of_words(vocabulary,sentence):
    #construct bag of words
    l1 = []
    for w in vocabulary:
        if w in sentence:
            l1.append(1)
        else:
            l1.append(0)
    return l1

def get_weighted_word_vecs(vocabulary,sentence,tokens):
    a = 0.001
    sum_array = [0]*300
    for w in vocabulary:
        if w in sentence:
            v = nlp(w)
            v_vec = v.vector #get the word2vec vectors here
            count_word = tokens.count(w)
            weight = a/(a+count_word)
            sum_array = sum_array+ (v_vec*weight)
    return sum_array

class Features:
    def __init__(self,df):
        self.df = df
        self.data = pd.DataFrame()
    
    def generate(self):
        self.data["cosine"] = self.df.apply(self.__cosine_simlarity__,axis=1)
        self.data["label"] = self.df.apply(self.__get_gold_tag__,axis=1)
        self.data["word_vectors"] = self.df.apply(self.__word_to_vec__,axis=1)
        return self.data
           
    def __get_gold_tag__(self,row):
        return row['Gold Tag']

    def __cosine_simlarity__(self,row):
        # cosine formula
        sentences_spacified = Preprocessing().__spacifyText__(row)
        s1 = sentences_spacified["corpus"][0]
        s2=  sentences_spacified["corpus"][0]
        print(s1,s2)

        v1 = bag_of_words(row['vocabulary'],sentFeatures["lemmas"])
        v2 = bag_of_words(row['vocabulary'],sentFeatures["lemmas"])
        c = 0
        c1 = 0
        c2 = 0
        for i in range(len(v1)):
            c += v1[i]*v2[i]
            c1 += v1[i]**2
            c2 += v2[i]**2
        cosine = c / float((c1*c2)**0.5)
        return cosine

    def __word_to_vec__(self,row):
        sentFeatures = Preprocessing(row)
        v1 = get_weighted_word_vecs(row['vocabulary'],row["Sentence1"],row["tokens"])
        v2 = get_weighted_word_vecs(row['vocabulary'],row["Sentence2"],row["tokens"])

        c = 0
        c1 = 0
        c2 = 0
        for i in range(len(v1)):
            c += v1[i]*v2[i]
            c1 += v1[i]*v1[i]
            c2 += v2[i]*v2[i]
        
        weighted_cosine = c / float((c1*c2)**0.5)
        return weighted_cosine

