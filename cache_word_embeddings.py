# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'

from wordembedding import WordEmbedding
from utils import Utils

import pickle

"""
Prerequisites:
- you ran the preprocessing at least once to have the cached processed files in the process_data directory
- you need to put the glove pre-trained model file in this folder.
- set the name of the 'glove_file' variable to the corresponding file
- see here if you need the file: https://nlp.stanford.edu/projects/glove/
"""

glove_file = 'glove.840B.300d.txt'
paragraph_ids_file = 'process_data/paragraph_ids.pkl'
processed_paragraph_file = 'process_data/lemma_processed_paragraph.pkl'
test_file = 'process_data/process_test.pkl'


def cache_terms_embedding_vectors():
    print('Load Data...')

    # Load Data to create the count matrix
    paragraph_ids = pickle.load(open(paragraph_ids_file, 'rb'))
    corpus = pickle.load(open(processed_paragraph_file, 'rb'))
    test = pickle.load(open(test_file, 'rb'))

    # Here load the pre-trained glove model
    # Not included in the git, since it is 5GB big.
    model = WordEmbedding(glove_file)

    doc_structure = Utils.get_document_structure_from_data(test, corpus, paragraph_ids)

    print('Create Vocabulary...')
    # the vocabulary is a set of terms
    vocabulary = Utils.create_vocabulary_from_dict(doc_structure)
    print(len(vocabulary))

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
