from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
import pandas as pd
  
class Preprocessing:
    def __init__(self, data):
        self.data = data

    # concat s1 and s2 together
    def getAllSentences(self):
        s1 = self.data[['Sentence1']]
        s2 = self.data[['Sentence2']]
        s1.columns = ['Sentence']
        s2.columns = ['Sentence']
        s = pd.concat([s1,s2], ignore_index=True)
        return s
        
    
    def tokenize(self, sentence):
        #TODO
        hi

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
                
