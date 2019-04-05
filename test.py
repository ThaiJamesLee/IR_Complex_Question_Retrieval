# -*- coding: utf-8 -*-
# this is used to test implementations
# and to show how to use the implemented code
import pickle
import numpy as np


from bm25 import BM25
from tf_idf import TFIDF
from similarity import Similarity


from utils import Utils

##################
# Load test and processed data
##################
print('================== Load Data ===================')
# contains list of query strings. querty terms are separated by whitespace
queries = pickle.load(open('processed_data/processed_query.pkl', 'rb'))
# load paragraph corpus that contains list of strings separated by whitespace.
corpus = pickle.load(open('processed_data/processed_paragraph.pkl', 'rb'))
test = pickle.load(open('processed_data/simulated_test.pkl', 'rb'))
paragraph_ids = pickle.load(open('processed_data/paragraph_ids.pkl', 'rb'))
unique_query = np.unique(list(test['query']))
# y_true = pickle.load(open('processed_data/y_true.pkl', 'rb'))
# print(y_true.columns)
unique_doc = np.unique(list(test['docid']))

c_corpus = [(id, corpus[paragraph_ids.index(id)].split()) for id in unique_doc]
doc_structure = dict()
# retrieve query at row 0, col 0
raw_query = pickle.load(open('processed_data/raw_query.pkl', 'rb'))
# for idx, query in enumerate(raw_query):
#    print(query[7:].replace("/", " ").replace("%20", " "))

print(test.iloc[0, 0][7:].replace("/", " ").replace("%20", " "))
print(Utils.get_doc_and_relevance_by_query(test.iloc[0, 0], test))
for (id, doc) in c_corpus:
    value = {word: doc.count(word) for word in doc}
    doc_structure.update({id: value})
# print(doc_structure)

##################
# BM25
##################


print('================== BM25 Test ===================')

bm25 = BM25(doc_structure, k=1.2)

print(bm25.get_term_frequency_in_doc('ddddd', 'f8b6fc8f2c326f1f25a65216a58b426910e523c6'))

print(bm25.idf_weight('they'))

print(bm25.relevance('f8b6fc8f2c326f1f25a65216a58b426910e523c6', 'they'))

relevance_scores = bm25.compute_relevance_on_corpus('dimension hidden imag display')
print('Relevance Score Test')
print(relevance_scores)

# sort dict by value
# https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value?page=1&tab=votes#tab-top
for w in sorted(relevance_scores, key=relevance_scores.get, reverse=True):
  print(w, relevance_scores[w])

# print('Test Utils create vocabulary')
# vocabulary = Utils.create_vocabulary_from_dict(doc_structure)
# print(vocabulary)
# print(len(vocabulary))

##################
#
##################
print('================== TF_IDF Test ===================')

tf_idf = TFIDF(doc_structure)

tf_idf_matrix = tf_idf.term_doc_matrix
print('TF IDF Matrix')
# print(tf_idf_matrix)

query_vector = tf_idf.create_query_vector(queries[40])
print('Example Query Vector with Q='+queries[40])
print(query_vector)

##################
# Cosine Similarity
##################

print('================== Cosine Similarity Test ===================')
print('Compare with Query='+queries[40])

similarity_scores = {}
for docs, terms in tf_idf_matrix.items():
    score = Similarity.cosine_similarity(terms, query_vector)
    similarity_scores.update({docs: score})
print('Cosine Similarity between query and set docs in tf idf matrix')
print('Query Vector:')
print(query_vector)
print(similarity_scores)

print('================== Performance Test ===================')

