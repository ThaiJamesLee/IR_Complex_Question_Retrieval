# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'

from wordembedding import WordEmbedding
from utils import Utils

import pickle
import numpy as np

print('Load Data...')

# Load Data to create the count matrix
paragraph_ids = pickle.load(open('processed_data/paragraph_ids.pkl', 'rb'))
corpus = pickle.load(open('processed_data/processed_paragraph.pkl', 'rb'))
test = pickle.load(open('processed_data/simulated_test.pkl', 'rb'))

# Here load the pre-trained glove model
# Not included in the git, since it is 5GB big.
model = WordEmbedding('glove.840B.300d.txt')

print('Prepare Data Structures...')
unique_doc = np.unique(list(test['docid']))
c_corpus = [(id, corpus[paragraph_ids.index(id)].split()) for id in unique_doc]
doc_structure = dict()

# create the count matrix
for (idx, doc) in c_corpus:
    value = {word: doc.count(word) for word in doc}
    doc_structure.update({idx: value})

print('Create Vocabulary...')
# the vocabulary is a set of terms
vocabulary = Utils.create_vocabulary_from_dict(doc_structure)
print(vocabulary)

term_embedding_vector = {}

# we iterate over all terms in the vocabulary and get the word embedding vector
# we save them in a dict as {term: vector, ...}
for term in vocabulary:
    try:
        print('Save vector of term: ', term)
        term_embedding_vector.update({term: model.get_word_vector_2(term)})
    except KeyError:
        pass

print('Dump dict of term and embedding vector in the cache...')
# save the vectors in cache so we don't need to load the whole pre-trained model
pickle.dump(term_embedding_vector, open('cache/word_embedding_vectors.pkl', 'wb'))
print('Finish saving.')
