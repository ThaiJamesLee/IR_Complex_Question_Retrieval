# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'
"""
this is used to test implementations
and to show how to use the implemented code
"""
import pickle
import numpy as np
import string
import re

from bm25 import BM25
from tf_idf import TFIDF
from similarity import Similarity
from performance import Performance
from performance import AveragePrecision
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import trec_car.read_data

from utils import Utils
from preprocessing import Preprocess

#################
# Stemmer setup
#################
stemmer = PorterStemmer()
new_punctuation = np.array(string.punctuation)
regex = re.compile(f'[{re.escape(string.punctuation)}]')
stopword = set(stopwords.words("english"))


##################
# Load test and processed data
##################
# print(len(pickle.load(open('cache/avg_emb_vec_glove_840B_300d.pkl', 'rb'))))

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
query_map = {}

index = 0
N = 50
for idx, query in enumerate(raw_query):
    if index < N:
        prep = Preprocess.preprocess('stem',[query[7:].replace("/", " ").replace("%20", " ")])
        query_map.update({query: prep[0]})
        index += 1
    else:
        break

# print(query_map)
# print(test.iloc[0, 0][7:].replace("/", " ").replace("%20", " "))
# test is a dataframe
# print(Utils.get_doc_and_relevance_by_query(test.iloc[0, 0], test))
for (id, doc) in c_corpus:
    value = {word: doc.count(word) for word in doc}
    doc_structure.update({id: value})
print(doc_structure)

##################
# BM25
##################


print('================== BM25 Test ===================')

bm25 = BM25(doc_structure, k=1.2)

print(bm25.get_term_frequency_in_doc('ddddd', 'f8b6fc8f2c326f1f25a65216a58b426910e523c6'))

print(bm25.idf_weight('they'))

print(bm25.relevance('f8b6fc8f2c326f1f25a65216a58b426910e523c6', 'they'))

rel_scores = {}
idx = 0
# iterate over queries in query map
# calculate for every document the relevance score for given query
for k, v in query_map.items():
    scores = bm25.compute_relevance_on_corpus(v)
    rel_scores.update({k: scores})

# output is too big, just uncomment if you want to check the output
print(rel_scores)

# relevance_scores = bm25.compute_relevance_on_corpus('dimension hidden imag display')
# print('Relevance Score Test')
# print(relevance_scores)

# sort dict by value
# https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value?page=1&tab=votes#tab-top
# for w in sorted(relevance_scores, key=relevance_scores.get, reverse=True):
#   print(w, relevance_scores[w])

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

# output is too big, just uncomment if you want to check the output
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

# output is too big, just uncomment if you want to check the output
# print(similarity_scores)

print('================== Performance Test ===================')

# k = unprocessed query
# v = processed query
p = AveragePrecision()
for k, v in query_map.items():
    # get docids and relevance columns for given query as dataframe
    # this is y_true
    # test is dataframe from simulated_test.pkl
    y_true = Utils.get_doc_and_relevance_by_query(k, test)
    # rel_scores = {query : {docid: score}}
    y_pred = rel_scores[k]
    print(v)
    precision = AveragePrecision.avg_precision_score(y_pred, y_true)
    print('Avg. Precision Score', precision)
    recall = AveragePrecision.avg_recall_score(y_pred, y_true)
    print('Avg. Recall Score', recall)
    p.add_precision_score(v, precision)
    p.add_recall_score(v, recall)
print(p.mean_avg_precision())

