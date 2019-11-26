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
from nltk.wsd import lesk
from collections import defaultdict
from spacy.lang.en.stop_words import STOP_WORDS as eng_stopwords
import string
sp = spacy.load('en_core_web_sm')
class Lemmas:
    def __init__(self,word,tag):
        self.text=word
        self.tag_=tag
    def __repr__(self):
        return "{}|{}".format(self.text,self.tag_)
def get_synsets(word):
        word_synset = wordnet.synsets(word)
        return word_synset

def get_wordnet_pos(treebank_tag):
    if treebank_tag is None:
        return wordnet.NOUN
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
        sentArray = [row["Sentence1"].lower(),row["Sentence2"].lower()]
        for i,sent in enumerate(sentArray):
            sentArray[i] = sp(sent)
        return sentArray
    def __tokenizer_spacy__(self,row):
        """
        Generate tokens for corpus
        """
        corpus= row['corpus']
        tokens =[]
        for sent in corpus:
            tokens.append([Lemmas(token.text,token.tag_) for token in sent])
        # tokens = [x for sublist in tokens for x in sublist]
        return tokens
    def __tokenizer_spacy_filter__(self,row):
        """
        Remove stop words from tokens
        """
        tokens= row['tokens']
        output=[]
        for sent in tokens:
            output.append([x for x in sent if x.text not in eng_stopwords and x.text not in string.punctuation])
        return output
    def __pos_spacy__(self,row):
        """
        Generate pos tags for tokens
        """
        tokens = row["tokens"]
        output=[]
        for sent in tokens:
            output.append( [(x.text,x.tag_) for x in sent])
        return output
    def __pos_spacy_filter__(self,row):
        """
        Generate pos tags for filtered tokens
        """
        pos_tagged= row['pos_tagged']
        output = []
        for sent in pos_tagged:
            output.append([x for x in sent if x[0] not in eng_stopwords and x[0] not in string.punctuation])
        return output
    def __lemmatize__(self, row):
        """
        Generate lemma for tokens
        """
        tokens_filtered = row["tokens_filtered"]
        output = []
        lemmatizer = WordNetLemmatizer()
        for sent in tokens_filtered:
            lemmas =[]
            for word in sent:
                tag = get_wordnet_pos(word.tag_)
                if tag is  None:
                    w = lemmatizer.lemmatize(word.text)
                else:
                    w = lemmatizer.lemmatize(word.text,tag)
                lemmas.append(Lemmas(w,tag))
            output.append(lemmas)
        return output
    def __hypernyms__(self,row):
        """
        Generate hypernyms for tokens
        """
        hypernyms = defaultdict(list)
        lemmas = row['lemmas']
        for sent in lemmas:
            for word in sent:
                hypernyms[word.text]=list(get_hypernyms(word.text))
        return hypernyms
    def __hyponyms__(self,row):
        """
        Generate hyponyms for tokens
        """
        lemmas = row['lemmas']
        hyponyms = defaultdict(list)
        for sent in lemmas:
            for word in sent:
                hyponyms[word.text]=list(get_hyponyms(word.text))
        return hyponyms
    def __meronyms__(self,row):
        """
        Generate meronyms for tokens
        """
        lemmas = row['lemmas']
        meronyms = defaultdict(list)
        for sent in lemmas:
            for word in sent:
                meronyms[word.text]=list(get_meronyms(word.text))
        return meronyms
    def __holonyms__(self,row):
        """
        Generate holonyms for tokens
        """
        lemmas = row['lemmas']
        holonyms = defaultdict(list)
        for sent in lemmas:
            for word in sent:
                holonyms[word.text]=list(get_holonyms(word.text))
        return holonyms
    def __generateParseTree__(self,row):
        """
        Generate parsetree for sentences
        """
        corpus = row['corpus']
        output  = []
        for r in corpus:
            output.append([to_nltk_tree(sent.root) for sent in r.sents][0])
        return output

    def __wordnet_lesk_wsd__(self,row):
        """
        Word sense disambugation 
        """
        lemmas = row["tokens"]
        nv_dict= defaultdict(lambda: defaultdict(lambda: []))
        # nv_dict = {0:{"n":[],"v":[]},1:{"n":[],"v":[]}}
        for i,x in enumerate(lemmas):
            for y in x:
                tag = get_wordnet_pos(y.tag_)
                if(tag in ["n","v"]):
                    wsd = lesk(x,y.text,tag)
                    nv_dict[i][tag].append((y.text,wsd))
        return nv_dict

    def __get_vocab_from_lemmas_set(self,row):
        lemmas = row["lemmas"]
        output = set()
        for sent in lemmas:
            for word in sent:
                output.add(word.text)
        return output
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
        self.data['vocabulary'] = self.data.apply(self.__get_vocab_from_lemmas_set,axis=1)
        self.data["lesk_wsd"] = self.data.apply(self.__wordnet_lesk_wsd__,axis=1)
        return self
        

    def store(self,name):
        """
        Store preprocessed data for reuse
        """
        # print(self.data.loc[0])
        file_path = "../PreProcessesData/{}".format(name)
        directory = os.path.dirname(file_path)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)  
        self.data.to_pickle(file_path)