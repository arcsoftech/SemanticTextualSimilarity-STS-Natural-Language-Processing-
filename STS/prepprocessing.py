from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
from nltk.corpus import wordnet
import spacy
sp = spacy.load('en_core_web_sm')

class Preprocessing:
    def __init__(self, data):
        self.data = data
        self.output = {}
        self.generateCorpus()

    def tokenizer(self, sentence):
        words = word_tokenize(sentence)
        return words

    

    def remove_stopwords(self, words):
        eng_stopwords = stopwords.words('english')
        filtered_words = {w for w in words if not w in eng_stopwords}
        return filtered_words

    def pos(self, tokenized_words):
        pos_tagged = nltk.pos_tag(tokenized_words)
        return pos_tagged
    def pos_spacy(self,sentArray):
        for sent in sentArray:
            sen = sp(sent)
            for x in sen:
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
            word_synset = wn.synsets(word)
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

    def generateCorpus(self):
        for index, row in self.data.iterrows():
            corpus = [row["Sentence1"],row["Sentence2"]]
            tokens = [self.tokenizer(s) for s in corpus]
            tokens = [x for sublist in tokens for x in sublist]
            tokens_filtered = self.remove_stopwords(tokens)
            posTaggedWords = self.pos(tokens_filtered)
            posTaggedWords_spacy = self.pos_spacy(corpus)
            # lemmas = self.lemmatize(tokens,posTaggedWords)
            tuple_filter = lambda t, i, w: filter(lambda a: a[0] == w)
            newtuple = tuple_filter(posTaggedWords, 0, 'leaders')
            print(newtuple)
            # print(posTaggedWords_spacy)
            return
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

