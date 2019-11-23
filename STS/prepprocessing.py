from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
from nltk.corpus import wordnet
import spacy
import networkx as nx
from nltk import Tree

sp = spacy.load('en_core_web_sm')
eng_stopwords = stopwords.words('english')

class Preprocessing:
    def __init__(self, data):
        self.data = data
        self.output = {}
        self.generateCorpus()

    def tokenizer(self, sentence):
        words = word_tokenize(sentence)
        return words  

    def remove_stopwords(self, words):
        
        filtered_words = {w for w in words if not w in eng_stopwords}
        return filtered_words

    # def pos(self, tokenized_words):
    #     pos_tagged = nltk.pos_tag(tokenized_words)
    #     return pos_tagged
    
    def spacifyText(self,sentArray):
        for sent in sentArray:
            yield sp(sent)
    def pos_spacy(self,sentArray):
        for x in sentArray:
            yield (x.text,x.tag_)
    def lemmatize(self, words,posTaggedWords):
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
                return ''
        lemmatizer = WordNetLemmatizer()
        for word in words:
            tag = posTaggedWords.get(word)
            if tag == "":
                w = lemmatizer.lemmatize(word)
            else:
                w = lemmatizer.lemmatize(word,get_wordnet_pos(tag))
            yield w

    def get_synsets(self, word):
            word_synset = wordnet.synsets(word)
            return word_synset

    def get_hypernymns(self, word):
            word_synsets = self.get_synsets(word)
            for synset in word_synsets:
                for s in synset.hypernyms():
                    for t in s.lemmas():
                        yield t.name()

    def get_hyponymns(self, word):
        word_synsets = self.get_synsets(word)
        word_hyponymns = []
        for synset in word_synsets:
            for s in synset.hyponymns():
                for t in s.lemmas():
                    yield t.name()

    def get_meronyms(self, word):
        word_synsets = self.get_synsets(word)
        for synset in word_synsets:
            for s in synset.part_meronyms():
                for t in s.lemmas():
                    yield t.name()
            for s in synset.substance_meronyms():
                for t in s.lemmas():
                    yield t.name()

    def get_holonyms(self, word):
        word_synsets = self.get_synsets(word)
        word_holonyms = []
        for synset in word_synsets:
            for s in synset.part_holonyms():
                for t in s.lemmas():
                    yield t.name()
            for s in synset.substance_holonyms():
                for t in s.lemmas():
                    yield t.name()
    
    def tok_format(self,tok):
        return "_".join([tok.orth_, tok.tag_, tok.dep_])
    
    def to_nltk_tree(self,node):
        if node.n_lefts + node.n_rights > 0:
            return Tree(self.tok_format(node), [self.to_nltk_tree(child) for child in node.children])
        else:
            return self.tok_format(node)

    def generateParseTree(self,row):
        corpus = [row["Sentence1"],row["Sentence2"]]
        for r in corpus:
            command = r
            en_doc = sp(u'' + command) 
            [self.to_nltk_tree(sent.root).pretty_print() for sent in en_doc.sents]

    def generateCorpus(self):
        self.data = self.data.reindex(self.data.columns.tolist() + ['corpus','posTaggedWords'])
        print(self.data)
        # for index, row in self.data.iterrows():
        #     corpus = [row["Sentence1"],row["Sentence2"]]
        #     print(corpus)
            # spacified_corpus= list(self.spacifyText(corpus))
            # print(spacified_corpus)
            # self.data.insert(index, "corpus", spacified_corpus, True) 
            # posTaggedWords = self.pos_spacy(spacified_corpus)
            # df.insert(index, "posTaggedWords", posTaggedWords, True) 
            # tokens = [token.text for token in spacified_corpus]
            # df.insert(index, "tokens", tokens, True) 
            # tokens_filtered = [x for x in tokens if x not in eng_stopwords]
            # df.insert(index, "tokens_filtered", tokens_filtered, True) 
            # posTaggedWords_filtered = [x for x in posTaggedWords if x[0] not in eng_stopwords]
            # df.insert(index, "posTaggedWords_filtered", posTaggedWords_filtered, True) 

            

            # tokens = [self.tokenizer(s) for s in corpus]
            # tokens = [x for sublist in tokens for x in sublist]
            # tokens_filtered = self.remove_stopwords(tokens_filtered)
            # posTaggedWords = self.pos(tokens)
            # posTaggedWords_spacy = self.pos_spacy(corpus)
            # lemmas = self.lemmatize(tokens,posTaggedWords)
            # tuple_filter = lambda t, i, w: filter(lambda a: a[0] == w)
            # newtuple = tuple_filter(posTaggedWords, 0, 'leaders')
            # print(newtuple)
            # print(self.data.loc[index])
            # return
            # self.vocabulary = list(set(lemmas))
            # self.corpus = tokens
            # hypernyms = {word: list(self.get_hypernymns(word)) for word in tokens}
            # meronyms = {word: list(self.get_meronyms(word)) for word in tokens}
            # holonyms = {word: list(self.get_holonyms(word)) for word in tokens}
            
            # self.output.update({"tokens": tokens, "stop_word_remoevd": tokens_filtered, "lemmas": lemmas,
            #                     "hypernyms": hypernyms, "meronyms": meronyms, "holonyms": holonyms, "pos": posTaggedWords})
            
        # sentences = self.getAllSentences()
        # tokens = [self.tokenize(s) for s in sentences]
        # tokens = [x for sublist in tokens for x in sublist]
        # tokens_filtered = self.remove_stopwords(tokens)
        # lemmas = self.lemmatize(tokens)
        # self.vocabulary = list(set(lemmas))
        # self.corpus = tokens
        # hypernyms = {word: list(self.get_hypernymns(word)) for word in tokens}
        # meronyms = {word: list(self.get_meronyms(word)) for word in tokens}
        # holonyms = {word: list(self.get_holonyms(word)) for word in tokens}
        # posTaggedWords = self.pos(tokens)
        # self.output.update({"tokens": tokens, "stop_word_remoevd": tokens_filtered, "lemmas": lemmas,
        #                     "hypernyms": hypernyms, "meronyms": meronyms, "holonyms": holonyms, "pos": posTaggedWords})

