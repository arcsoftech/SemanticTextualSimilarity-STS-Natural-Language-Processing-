import pandas as pd
import spacy
import gensim.downloader as api
word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data
from nltk.corpus import wordnet as wn
import os

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
    syn1= wn.synset(syn1)
    syn2= wn.synset(syn2)
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
    return avg
    # if 0<=avg<=0.2:
    #     return 1
    # if 0.2<=avg<=0.4:
    #     return 2
    # if 0.4<=avg<=0.6:
    #     return 3
    # if 0.6<=avg<=0.8:
    #     return 4
    # else:
    #     return 5

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
    
    def __get_gold_tag__(self,row):
        try:
            return row['Gold Tag']
        except:
            return None

    def __cosine_simlarity__(self,row):
        # cosine formula
        lemmas1 = []
        for word in row["lemmas"][0]:
            lemmas1.append(word.text)
        
        lemmas2 = []
        for word in row["lemmas"][1]:
            lemmas2.append(word.text)

        v1 = bag_of_words(row['vocabulary'],lemmas1)
        v2 = bag_of_words(row['vocabulary'],lemmas2)
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

        v1 = get_weighted_word_vecs(row['vocabulary'],lemmas1,lemmas1+lemmas2)
        v2 = get_weighted_word_vecs(row['vocabulary'],lemmas2,lemmas1+lemmas2)

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
        noun_list1 = dict1[0].n
        verb_list1 = dict1[0].v

        noun_list2 = dict1[1].n
        verb_list2 = dict1[1].v

        nouns_scores_list = get_pairs_similarity(noun_list1,noun_list2)
        verbs_scores_list = get_pairs_similarity(verb_list1,verb_list2)

        final_score = get_final_similarity_score(nouns_scores_list,verbs_scores_list)

        return final_score
    def __get_context_overap__(self,row):
        lemmas = row["lemmas"]
        hypernyms = row["hypernyms"]
        hyponyms = row["hyponyms"]
        synonyms = row["synonyms"]
        sentA =set()
        sentB= set()
        for i in range(len(lemmas)):
            if(i==0):
                sentA.update(lemmas[i])
                sentA.update([x for sublist in hypernyms[i].values() for x in sublist])
                sentA.update([x for sublist in hyponyms[i].values() for x in sublist])
                sentA.update([x for sublist in synonyms[i].values() for x in sublist])
            else:
                sentB.update(lemmas[i])
                sentB.update([x for sublist in hypernyms[i].values() for x in sublist])
                sentB.update([x for sublist in hyponyms[i].values() for x in sublist])
                sentA.update([x for sublist in synonyms[i].values() for x in sublist])
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
        tokens = row["lemmas"]
        sentence1=[x.text for x in tokens[0]]
        sentence2=[x.text for x in tokens[1]]
        similarity = word_vectors.wmdistance(sentence1, sentence2)
        return similarity

    def generate(self):
        self.data["cosine"] = self.df.apply(self.__cosine_simlarity__,axis=1)
        self.data["word_vectors"] = self.df.apply(self.__word_to_vec__,axis=1)
        self.data['heuristic_similarity'] = self.df.apply(self.__get_heuristic_similarity__,axis=1)
        self.data["conceptOverlap"] = self.df.apply(self.__get_context_overap__,axis=1)
        self.data["jaccard"] = self.df.apply(self.__jaccard_similarity__,axis=1)
        self.data["wmd"] = self.df.apply(self.__wmd__,axis=1)
        self.data["label"] = self.df.apply(self.__get_gold_tag__,axis=1)
        return self
    def store(self,file_path):
        """
        Store feature data for reuse
        """
        print(self.data.loc[0])
        directory = os.path.dirname(file_path)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)  
        self.data.to_pickle(file_path)
           
  
