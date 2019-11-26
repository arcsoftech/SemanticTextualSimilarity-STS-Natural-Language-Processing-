import pandas as pd
import spacy

import gensim.downloader as api
word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data

nlp = spacy.load("en_core_web_lg")
def bag_of_words(vocabulary,sentence):
    #construct bag of words
    l1 = []
    sentence= [x.text for x in sentence]
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
    def __get_gold_tag__(self,row):
        return row['Gold Tag']

    def __cosine_simlarity__(self,row):
        # cosine formula
        v1 = bag_of_words(row['vocabulary'],row["lemmas"][0])
        v2 = bag_of_words(row['vocabulary'],row["lemmas"][1])
        c = 0
        c1 = 0
        c2 = 0
        for i in range(len(v1)):
            c += v1[i]*v2[i]
            c1 += v1[i]**2
            c2 += v2[i]**2
        try:
            cosine = c / float((c1*c2)**0.5)
        except:
            return 0
        return cosine

    def __word_to_vec__(self,row):
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
    def __get_context_overap__(self,row):
        lemmas = row["lemmas"]
        hypernyms = row["hypernyms"]
        hyponyms = row["hyponyms"]
        holonyms = row["holonyms"]
        meronyms = row["meronyms"]
        sentA =set()
        sentB= set()
        for i in range(len(lemmas)):
            if(i==0):
                sentA.update(lemmas[i])
                sentA.update([x for sublist in hypernyms[i].values() for x in sublist])
                sentA.update([x for sublist in hyponyms[i].values() for x in sublist])
                sentA.update([x for sublist in holonyms[i].values() for x in sublist])
                sentA.update([x for sublist in meronyms[i].values() for x in sublist])
            else:
                sentB.update(lemmas[i])
                sentB.update([x for sublist in hypernyms[i].values() for x in sublist])
                sentB.update([x for sublist in hyponyms[i].values() for x in sublist])
                sentB.update([x for sublist in holonyms[i].values() for x in sublist])
                sentB.update([x for sublist in meronyms[i].values() for x in sublist])
        intersection = sentA.intersection(sentB)
        union = sentA.union(sentB)

        return len(intersection)/len(union)

    def __jaccard_similarity__(self,row):
        tokens = row["tokens_filtered"]
        a = set([x.text for x in tokens[0]]) 
        b = set([x.text for x in tokens[1]])
        intersection = a.intersection(b)
        union = a.union(b)

        return len(intersection)/len(union)

    def __wmd__(self,row):
        """
        Word Movers disance
        Reference:https://radimrehurek.com/gensim/models/keyedvectors.html
        """
        tokens = row["tokens"]
        sentence1=[x.text for x in tokens[0]]
        sentence2=[x.text for x in tokens[0]]
        similarity = word_vectors.wmdistance(sentence1, sentence2)
        return similarity
    def generate(self):
        # self.data["cosine"] = self.df.apply(self.__cosine_simlarity__,axis=1)
        # self.data["word_vectors"] = self.df.apply(self.__word_to_vec__,axis=1)
        self.data["conceptOverlap"] = self.df.apply(self.__get_context_overap__,axis=1)
        self.data["jaccard"] = self.df.apply(self.__jaccard_similarity__,axis=1)
        self.data["wmd"] = self.df.apply(self.__wmd__,axis=1)
        self.data["label"] = self.df.apply(self.__get_gold_tag__,axis=1)
        return self.data
           
    
