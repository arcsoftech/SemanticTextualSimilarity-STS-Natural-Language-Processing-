from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
from nltk.corpus import wordnet
import spacy
import networkx as nx
from nltk import Tree
import os

sp = spacy.load('en_core_web_sm')
eng_stopwords = stopwords.words('english')
def get_synsets(word):
        word_synset = wordnet.synsets(word)
        return word_synset

def get_hypernyms(word):
    word_synsets = get_synsets(word)
    for synset in word_synsets:
        for s in synset.hypernyms():
            for t in s.lemmas():
                yield t.name()

def get_hyponyms(word):
    word_synsets = get_synsets(word)
    for synset in word_synsets:
        for s in synset.hyponyms():
            for t in s.lemmas():
                yield t.name()

def get_meronyms(word):
    word_synsets = get_synsets(word)
    for synset in word_synsets:
        for s in synset.part_meronyms():
            for t in s.lemmas():
                yield t.name()
        for s in synset.substance_meronyms():
            for t in s.lemmas():
                yield t.name()

def get_holonyms(word):
    word_synsets = get_synsets(word)
    for synset in word_synsets:
        for s in synset.part_holonyms():
            for t in s.lemmas():
                yield t.name()
        for s in synset.substance_holonyms():
            for t in s.lemmas():
                yield t.name()

def tok_format(tok):
    return "_".join([tok.orth_, tok.tag_, tok.dep_])

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)

class Preprocessing:
    def __init__(self, data):
        self.data = data
        self.output = {}
    
    def __spacifyText__(self,row):
        """
        Transform sentence using spacy english model
        """
        sentArray = [row["Sentence1"],row["Sentence2"]]
        for i,sent in enumerate(sentArray):
            sentArray[i] = sp(sent)
        return sentArray
    def __pos_spacy__(self,row):
        """
        Generate pos tags for tokens
        """
        tokens = row["tokens"]
        for i,x in enumerate(tokens):
            tokens[i]= (x.text,x.tag_)
        return tokens
    def __lemmatize__(self, row):
        """
        Generate lemma for tokens
        """
        tokens_filtered = row["tokens_filtered"]
        def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            else:
                return None
        lemmatizer = WordNetLemmatizer()
        for i,word in enumerate(tokens_filtered):
            tag = get_wordnet_pos(word.tag_)
            if tag is  None:
                w = lemmatizer.lemmatize(word.text)
            else:
                w = lemmatizer.lemmatize(word.text,tag)
            tokens_filtered[i] = w
        return tokens_filtered
    def __tokenizer_spacy__(self,row):
        """
        Generate tokens for corpus
        """
        corpus= row['corpus']
        tokens =[]
        for sent in corpus:
            tokens.append([token for token in sent])
        tokens = [x for sublist in tokens for x in sublist]
        return tokens
    def __tokenizer_spacy_filter__(self,row):
        """
        Remove stop words from tokens
        """
        tokens= row['tokens']
        return [x for x in tokens if x.text not in eng_stopwords]
    def __pos_spacy_filter__(self,row):
        """
        Generate pos tags for filtered tokens
        """
        pos_tagged= row['pos_tagged']
        return [x for x in pos_tagged if x[0] not in eng_stopwords]
    def __hypernyms__(self,row):
        """
        Generate hypernyms for tokens
        """
        lemmas = row['lemmas']
        hypernyms = {word: list(get_hypernyms(word)) for word in lemmas}
        return hypernyms
    def __hyponyms__(self,row):
        """
        Generate hyponyms for tokens
        """
        lemmas = row['lemmas']
        hyponyms = {word: list(get_hyponyms(word)) for word in lemmas}
        return hyponyms
    def __meronyms__(self,row):
        """
        Generate meronyms for tokens
        """
        lemmas = row['lemmas']
        meronyms = {word: list(get_meronyms(word)) for word in lemmas}
        return meronyms
    def __holonyms__(self,row):
        """
        Generate holonyms for tokens
        """
        lemmas = row['lemmas']
        holonyms = {word: list(get_holonyms(word)) for word in lemmas}
        return holonyms
    def __generateParseTree__(self,row):
        """
        Generate parsetree for sentences
        """
        corpus = row['corpus']
        for i,r in enumerate(corpus):
            corpus[i]=[to_nltk_tree(sent.root) for sent in r.sents][0]
        return corpus

  
    def transform(self):
        """
        Tranform given data to preprocessed text
        """
        self.data["corpus"] = self.data.apply(self.__spacifyText__,axis=1)
        self.data["tokens"] = self.data.apply(self.__tokenizer_spacy__,axis=1)
        self.data["tokens_filtered"] = self.data.apply(self.__tokenizer_spacy_filter__,axis=1)
        self.data["pos_tagged"] = self.data.apply(self.__pos_spacy__,axis=1)
        self.data["pos_tagged_filtered"] = self.data.apply(self.__pos_spacy_filter__,axis=1)
        self.data["lemmas"] = self.data.apply(self.__lemmatize__,axis=1)
        self.data["hypernyms"] = self.data.apply(self.__hypernyms__,axis=1)
        self.data["hyponyms"] = self.data.apply(self.__hyponyms__,axis=1)
        self.data["holonyms"] = self.data.apply(self.__holonyms__,axis=1)
        self.data["meronyms"] = self.data.apply(self.__meronyms__,axis=1)
        self.data['dependency_tree'] = self.data.apply(self.__generateParseTree__,axis=1)
        self.data['vocabulary'] = self.data.apply(lambda x:list(set(x["lemmas"])),axis=1)
        return self
        

    def store(self,name):
        """
        Store preprocessed data for reuse
        """
        file_path = "../PreProcessesData/{}".format(name)
        directory = os.path.dirname(file_path)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)  
        self.data.to_pickle(file_path)