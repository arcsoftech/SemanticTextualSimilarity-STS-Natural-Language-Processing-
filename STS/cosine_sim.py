# """Name:Query Interpretor
# Author:Arihant Chhajed
# Description:This module is for calculation of
# semantic similarity using bag of word approach
# between two sentenses using nltk tools."""

# import string
# import os
# import sys
# import logging
# import re
# import urllib.parse

# import nltk
# from nltk.corpus import stopwords
# from gensim import corpora, models, similarities
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet
# from random import randint

# LEMMATIZER = WordNetLemmatizer()
# logging.basicConfig()
# KB = ""
# REMOVE_PUNCTUATION_MAP = dict((ord(char), None) for char in string.punctuation)
# DICT = ""
# CORPUS = ""
# MODEL = ""
# INDEX = ""
# """nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet') """



# def model_generator(sent1,sent2):
#     """
#     it generate model and dictionary files for provided KB
#     """
#     try:
#         global KB, DICT, MODEL, CORPUS, INDEX, DB
#         KB = list(DB['FaqKB'].find({"ServiceName": "PMIBOT"}, {
#                   "ServiceName": 0, "_id": 0, "LastModified": 0, "_url": 0}))[0]['KB']
#         data = KB['Questions']
#         texts = [tokenize(doc) for doc in data if tokenize(doc) is not None]
#         DICT = corpora.Dictionary(texts)
#         DICT.save("kb.dict")  # store the dictionary, for future reference
#         CORPUS = [DICT.doc2bow(text) for text in texts]
#         # store to disk, for later use
#         corpora.MmCorpus.serialize("kb.mm", CORPUS)
#         MODEL = models.LsiModel(CORPUS, id2word=DICT, num_topics=len(DICT))
#         INDEX = similarities.MatrixSimilarity(
#             MODEL[CORPUS], num_features=len(DICT))
#         print("model generated succesfully")
#         return "model generated succesfully"
#     except:
#         return False


# def get_wordnet_pos(treebank_tag):
#     """
#     Wordnet tree bank
#     """
#     if treebank_tag.startswith('J'):
#         return wordnet.ADJ
#     elif treebank_tag.startswith('V'):
#         return wordnet.VERB
#     elif treebank_tag.startswith('N') and treebank_tag.startswith('NN'):
#         return wordnet.NOUN
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return ''


# def pos_tagger(tokens):
#     """
#     postagger to find the parts of speech of the sentence
#     """
#     postagged = nltk.pos_tag(tokens)
#     return postagged


# def lematization(word, pos):
#     """
#     Lematization is the process to convert word into its most related morphological form.
#     """
#     tag = get_wordnet_pos(pos)
#     if (tag is not ""):
#         return LEMMATIZER.lemmatize(word, pos=tag)
#     else:
#         return LEMMATIZER.lemmatize(word)


# def stop_word_filteration(tokens):
#     """
#     Filter commonly occured stop_words
#     """
#     stop_words = set(stopwords.words('english'))
#     filtered_sentence = [w for w in tokens if w not in stop_words]
#     return filtered_sentence


# def tokenize(sentence):
#     """
#     Tokenization
#     """
#     tokens = nltk.word_tokenize(
#         sentence.lower().translate(REMOVE_PUNCTUATION_MAP))
#     filtered_sentence = stop_word_filteration(tokens)
#     pos_tagged = pos_tagger(filtered_sentence)
#     lemma = []
#     for (w, t) in pos_tagged:
#         lemma.append(lematization(w, t))
#     # for w in filtered_sentence:
#     #     stemmed.append(stemming(w))
#     return lemma


# @APP.route('/question', methods=['POST'])
# def run():
#     """
#     Run method to run QueryInterpreter for given query
#     """
#     print("Request recived by QueryInterpretor")
#     response = {}

#     try:
#         query = request.json
#         input_query = expand_abbrevations(query['question'], ABVT_DICT)
#         print(input_query)
#         if len(input_query.split()) >= 3:
#             vec = DICT.doc2bow(tokenize(input_query))
#             sims = INDEX[MODEL[vec]]
#             ans = list(enumerate(sims))
#             __max__ = 0
#             max_index = -1
#             for (index, per) in ans:
#                 if(per > __max__):
#                     __max__ = per
#                     max_index = index
#             _threshold = 0.60
#             if __max__ > _threshold and max_index is not -1:
#                 response['status'] = 200
#                 response['answer'] = KB['Answers'][max_index]
#             else:
#                 response['status'] = 201
#                 response['answer'] = "fallback"
#             print("Confidence Score and index value is", __max__, max_index)
#             return jsonify(response), 200
#         else:
#             response['status'] = 201
#             response['answer'] = "fallback"
#             return jsonify(response), 200
#     except():
#         response['status'] = 422
#         error_msg = "An unexpected error as occured."
#         response['message'] = error_msg + sys.exc_info()
#         return jsonify(response)


# @APP.route('/updateModel', methods=['GET'])
# def reinitialize():
#     """
#     reiinitialize model
#     """
#     # model.add_documents(another_tfidf_corpus)
#     print("reinitialized model")
#     result = model_generator()
#     response = {}
#     if result is not None:
#         response['status'] = 200
#         response['answer'] = result
#         return jsonify(response), 200
#     else:
#         response['status'] = 422
#         response['answer'] = "An unexpected error as occured."
#         return jsonify(response), 422


# @APP.route('/updateAbbr', methods=['GET'])
# def reinitializeAbbrDict():
#     """
#     reiinitialize ABBR DICT
#     """
#     # model.add_documents(another_tfidf_corpus)
#     result = update_abbr_dict()
#     response = {}
#     if result:
#         response['status'] = 200
#         response['answer'] = "Update ABBR Dict succesfull."
#         return jsonify(response), 200
#     else:
#         response['status'] = 422
#         response['answer'] = "An unexpected error as occured.."
#         return jsonify(response), 422


# if __name__ == "__main__":
#     try:
#         print("Server started")
#         update_abbr_dict()
#         if (os.path.exists("kb.dict")):
#             DICT = corpora.Dictionary.load('kb.dict')
#             CORPUS = corpora.MmCorpus('kb.mm')
#             MODEL = models.LsiModel(CORPUS, id2word=DICT, num_topics=len(DICT))
#             print(CORPUS)
#             print(DICT)
#             INDEX = similarities.MatrixSimilarity(
#                 MODEL[CORPUS], num_features=len(DICT))
#             KB = list(DB['PMIBOT'].find({"ServiceName": "PMIBOT"}, {
#                       "ServiceName": 0, "_id": 0, "LastModified": 0, "_url": 0}))[0]['KB']
#             print("Used files generated from existing model")
#         else:
#             print("Creating file")
#             model_generator()
#         APP.run(host='0.0.0.0', port=4244)
#         print("app started at port 4244")
#     except():
#         print("An unexpected error occured")