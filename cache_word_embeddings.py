# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'

from wordembedding import WordEmbedding
from utils import Utils

import pickle
import numpy as np

"""
Prerequisites:
- you ran the preprocessing at least once to have the cached processed files in the process_data directory
- you need to put the glove pre-trained model file in this folder.
- set the name of the 'glove_file' variable to the corresponding file
- see here if you need the file: https://nlp.stanford.edu/projects/glove/
"""

glove_file = 'glove.840B.300d.txt'

print('Load Data...')

# Load Data to create the count matrix
paragraph_ids = pickle.load(open('process_data/paragraph_ids.pkl', 'rb'))
corpus = pickle.load(open('process_data/lemma_processed_paragraph.pkl', 'rb'))
test = pickle.load(open('process_data/simulated_test.pkl', 'rb'))

# Here load the pre-trained glove model
# Not included in the git, since it is 5GB big.
model = WordEmbedding(glove_file)

doc_structure = Utils.get_document_structure_from_data(test, corpus, paragraph_ids)

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
        print('Passed Term = ', term)
        pass

print('Dump dict of term and embedding vector in the cache...')
# save the vectors in cache so we don't need to load the whole pre-trained model
pickle.dump(term_embedding_vector, open('cache/word_embedding_vectors.pkl', 'wb'))
print('Finish saving.')
