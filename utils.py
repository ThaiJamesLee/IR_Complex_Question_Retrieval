# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'


# Utils class for helping functions
# like static functions that can be used globally
class Utils:

    # Helper to add nested dicts
    # add dict into a parent dict
    # matrix: the parent dict consist of docs as keys
    # key: value: child dict
    @staticmethod
    def add_ele_to_matrix(matrix, doc, key, value):
        if doc not in matrix:
            matrix.update({doc: {key: value}})
        else:
            matrix[doc].update({key: value})
        return matrix

    # corpus: list of strings
    # each string consist of terms separated by whitespace
    # returns set of terms
    @staticmethod
    def create_vocabulary(corpus):
        vocabulary = set()
        for para in corpus:
            terms = para.split()
            words_set = set(terms)
            vocabulary = vocabulary.union(words_set)
        return vocabulary

    # corpus: a dict containing mappings of:
    # {doc_id: {term: value, term: value, ...}, ...}
    # returns set of terms
    @staticmethod
    def create_vocabulary_from_dict(corpus):
        vocabulary = set()
        for doc_id, terms in corpus.items():
            words_set = set(terms)
            vocabulary = vocabulary.union(words_set)
        return vocabulary
