import pandas as pd
import spacy
from ..Preprocessing.featureGenerator import Preprocessing
from nltk.corpus import wordnet as wn

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

def get_weighted_word_vecs(vocabulary,lemmas,lemmas_both):
    a = 0.001
    sum_array = [0]*300
    for w in vocabulary:
        if w in lemmas:
            v = nlp(w)
            v_vec = v.vector #get the word2vec vectors here
            count_word = lemmas_both.count(w)
            weight = a/(a+count_word)
            sum_array = sum_array+ (v_vec*weight)
    return sum_array

def get_wup_similarity(syn1,syn2):
    return syn1.wup_similarity(syn2) 

def get_final_similarity_score(nouns_scores_list,verbs_scores_list):
    n=0
    for i in nouns_scores_list:
        n+=i
    v=0
    for i in verbs_scores_list:
        v+=i
    # find avg of scores here   
    avg = (n+v)/(len(nouns_scores_list)+len(verbs_scores_list))

    if 0<=avg<=0.2:
        return 1
    if 0.2<=avg<=0.4:
        return 2
    if 0.4<=avg<=0.6:
        return 3
    if 0.6<=avg<=0.8:
        return 4
    else:
        return 5

def get_pairs_similarity(list1,list2):
    sim_scores = []
    for l1 in list1:
        for l2 in list2:
            if l1[1] is not None and l2[1] is not None:
                sim_scores.append(get_wup_similarity(l1[1],l2[1]))

    return sim_scores
    

class Features:
    def __init__(self,df):
        self.df = df
        self.data = pd.DataFrame()
    
    def generate(self):
        self.data["cosine"] = self.df.apply(self.__cosine_simlarity__,axis=1)
        self.data["word_vectors"] = self.df.apply(self.__word_to_vec__,axis=1)
        self.data['heuristic_similarity'] = self.df.apply(self.__get_heuristic_similarity__,axis=1)
        self.data["label"] = self.df.apply(self.__get_gold_tag__,axis=1)
        return self.data
           
    def __get_gold_tag__(self,row):
        return row['Gold Tag']

    def __cosine_simlarity__(self,row):
        # cosine formula
        lemmas1 = []
        for word in row["lemmas"][0]:
            lemmas1.append(word.text)
        
        lemmas2 = []
        for word in row["lemmas"][1]:
            lemmas2.append(word.text)

        v1 = bag_of_words(row['vocabulary'][0],lemmas1)
        v2 = bag_of_words(row['vocabulary'][1],lemmas2)
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
        lemmas1 = []
        for word in row["lemmas"][0]:
            lemmas1.append(word.text)
        
        lemmas2 = []
        for word in row["lemmas"][1]:
            lemmas2.append(word.text)

        v1 = get_weighted_word_vecs(row['vocabulary'][0],lemmas1,lemmas1+lemmas2)
        v2 = get_weighted_word_vecs(row['vocabulary'][1],lemmas2,lemmas1+lemmas2)

        c = 0
        c1 = 0
        c2 = 0
        for i in range(len(v1)):
            c += v1[i]*v2[i]
            c1 += v1[i]*v1[i]
            c2 += v2[i]*v2[i]
        
        weighted_cosine = c / float((c1*c2)**0.5)
        return weighted_cosine
    
    def __get_heuristic_similarity__(self,row):
        dict1 = row['lesk_wsd']
        noun_list1 = dict1[0]['n']
        verb_list1 = dict1[0]['v']

        noun_list2 = dict1[1]['n']
        verb_list2 = dict1[2]['v']

        nouns_scores_list = get_pairs_similarity(noun_list1,noun_list2)
        verbs_scores_list = get_pairs_similarity(verb_list1,verb_list2)

        final_score = get_final_similarity_score(nouns_scores_list,verbs_scores_list)

        return final_score
