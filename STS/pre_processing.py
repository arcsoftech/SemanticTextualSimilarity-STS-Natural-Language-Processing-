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
        self.generateCorpus()

    # concat s1 and s2 together
    def getAllSentences(self):
        s1 = self.data['Sentence1']
        s2 = self.data['Sentence2']
        s = pd.concat([s1, s2], ignore_index=True).values.tolist()
        return s

    def tokenize(self, sentence):
        words = word_tokenize(sentence)
        return words

    def getFeatures(self):
        return self.output

    # inputs list of words and returns filtered words
    def remove_stopwords(self, words):
        eng_stopwords = stopwords.words('english')
        filtered_words = {w for w in words if not w in eng_stopwords}
        return filtered_words

    # inputs list of words and returns list of words
    def lemmatize(self, words):
        lemmatizer = WordNetLemmatizer()
        len_words = len(words)
        for i in range(len_words):
            w = lemmatizer.lemmatize(words[i])
            words[i] = w
        return words

    def pos(self, tokenized_words):
        pos_tagged = nltk.pos_tag(tokenized_words)
        return pos_tagged

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

    def sent_vector(self, sent):
        X = self.tokenize(sent)
        L=self.lemmatize(X)
        X=X+L
        # for x in X:
        #     X+=self.get_hypernymns(x)
        l1 = []
        for w in self.vocabulary:
            if w in X:
                l1.append(1)
            else:
                l1.append(0)
        return l1

    def generateCorpus(self):
        sentences = self.getAllSentences()
        tokens = [self.tokenize(s) for s in sentences]
        tokens = [x for sublist in tokens for x in sublist]
        tokens_filtered = self.remove_stopwords(tokens)
        lemmas = self.lemmatize(tokens)
        self.vocabulary = list(set(lemmas))
        self.corpus = tokens
        hypernyms = {word: list(self.get_hypernymns(word)) for word in tokens}
        meronyms = {word: list(self.get_meronyms(word)) for word in tokens}
        holonyms = {word: list(self.get_holonyms(word)) for word in tokens}
        posTaggedWords = self.pos(tokens)
        self.output.update({"tokens": tokens, "stop_word_remoevd": tokens_filtered, "lemmas": lemmas,
                            "hypernyms": hypernyms, "meronyms": meronyms, "holonyms": holonyms, "pos": posTaggedWords})
