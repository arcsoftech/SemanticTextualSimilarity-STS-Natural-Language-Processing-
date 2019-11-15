from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
from nltk.corpus import wordnet as wn

class Preprocessing:
    def __init__(self, data):
        self.data = data
        self.output = {}

    # concat s1 and s2 together
    def getAllSentences(self):
        s1 = self.data[['Sentence1']]
        s2 = self.data[['Sentence2']]
        s1.columns = ['Sentence']
        s2.columns = ['Sentence']
        s = pd.concat([s1,s2], ignore_index=True)
        return s
        
    
    def tokenize(self, sentence):       
        words = word_tokenize(sentence)
        return words
        pass

    def getFeatures(self):
        pass

    # inputs list of words and returns filtered words 
    def remove_stopwords(self,words):
        eng_stopwords = stopwords.words('english')
        filtered_words = {w for w in words if not w in eng_stopwords}
        return filtered_words
    
    # inputs list of words and returns list of words
    def lemmatize(self,words):
        lemmatizer = WordNetLemmatizer()   
        len_words = len(words)
        for i in range(len_words):
            w = lemmatizer.lemmatize(words[i])
            words[i] = w
        return words

    def pos(self, tokenized_words):
        pos_tagged = nltk.pos_tag(tokenized_words)
        return pos_tagged

    def get_synsets(self,word):
        word_synset = wn.synsets(word)
        return word_synset
    
    def get_hypernymns(self,word):
        word_synsets = self.get_synsets(word)
        word_hypernyms =[]
        for synset in word_synsets:
            word_hypernyms.append(synset.hypernyms())
        return word_hypernyms
    
    def get_hyponymns(self,word):
        word_synsets = self.get_synsets(word)
        word_hyponymns =[]
        for synset in word_synsets:
            word_hyponymns.append(synset.hyponymns())
        return word_hyponymns
    