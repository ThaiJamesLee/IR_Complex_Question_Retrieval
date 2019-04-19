# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'

import pickle
import numpy as np
import string
import re

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from preprocessing import Preprocess


cached_embeddings = pickle.load(open('cache/word_embedding_vectors.pkl', 'rb'))
corpus = pickle.load(open('processed_data/processed_paragraph.pkl', 'rb'))
raw_corpus = pickle.load(open('processed_data/paragraphs.pkl', 'rb'))
corpusid = pickle.load(open('processed_data/paragraph_ids.pkl', 'rb'))

# dimension of the embedding vector
# we are currently using the pre-trained glove model
# glove.840.300d
vector_dimension = 300

#################
# Stemmer setup
#################
stemmer = PorterStemmer()
new_punctuation = np.array(string.punctuation)
regex = re.compile(f'[{re.escape(string.punctuation)}]')
stopword = set(stopwords.words("english"))

# create queries
raw_query = pickle.load(open('processed_data/raw_query.pkl', 'rb'))
query_map = {}

index = 0
N = 100
for idx, query in enumerate(raw_query):
    if index < N:
        prep = Preprocess.preprocess([query[7:].replace("/", " ").replace("%20", " ")])
        query_map.update({query: prep[0]})
        index += 1
    else:
        break

test = pickle.load(open('processed_data/simulated_test.pkl', 'rb'))
unique_doc = np.unique(list(test['docid']))
paragraph_ids = pickle.load(open('processed_data/paragraph_ids.pkl', 'rb'))

# create document corpus as dict
c_corpus = [(id, corpus[paragraph_ids.index(id)].split()) for id in unique_doc]
doc_structure = dict()
for (id, doc) in c_corpus:
    value = {word: doc.count(word) for word in doc}
    doc_structure.update({id: value})
# print(doc_structure)

# create cache for query average word embedding vector
query_embedding_vector = {}
for qu, qp in query_map.items():
    sum_embedding_vectors = np.zeros(vector_dimension)  # create initial empty array
    number_terms = len(query_map.keys())  # length of query
    terms = qp.split()

    for term in terms:
        try:
            sum_embedding_vectors = np.add(sum_embedding_vectors, cached_embeddings[term])
        except KeyError:
            # print('passed', term)
            pass

    for idx in range(vector_dimension):
        sum_embedding_vectors[idx] = sum_embedding_vectors[idx] / number_terms

    query_embedding_vector.update({qp: sum_embedding_vectors})

pickle.dump(query_embedding_vector, open('cache/avg_query_embeddings.pkl', 'wb'))

doc_embedding_vectors = {}
# create cache for document average word embedding vector
for docid, terms in doc_structure.items():
    sum_embedding_vectors = np.zeros(vector_dimension)
    number_terms = len(terms.keys())
    if number_terms > 0:
        for term in terms.keys():
            try:
                sum_embedding_vectors = np.add(sum_embedding_vectors, cached_embeddings[term])
            except KeyError:
                # print('passed', term)
                pass
        sum_embedding_vectors /= number_terms

    doc_embedding_vectors.update({docid: sum_embedding_vectors})
pickle.dump(query_embedding_vector, open('cache/avg_doc_embeddings.pkl', 'wb'))

print(pickle.load(open('cache/avg_doc_embeddings.pkl', 'rb')))








