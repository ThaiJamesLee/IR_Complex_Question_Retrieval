__author__ = 'Duc Tai Ly'
import math
import numpy as np
from utils import Utils


# Implementation based on dense vectors
class TFIDF:

    # documents: list of paragraphs
    # term_doc_matrix: contains tf_idf values for each term/doc entry
    def __init__(self, documents):
        self.documents = np.unique(documents)
        self.idf_vector = {}
        self.term_doc_matrix = self.create_tf_idf_matrix()

    # create from documents a term-doc matrix with occurrences of terms in corresponding doc
    def create_count_matrix(self):
        matrix = {}
        doc_id = 0
        for doc in self.documents:
            words = doc.split()
            words_set = set(words)
            word_dict = dict.fromkeys(words_set, 0)
            for word in words:
                word_dict[word] += 1
            doc = {str(doc_id): word_dict}
            matrix.update(doc)
            doc_id += 1
        return matrix

    # matrix: contains matrix of docs with terms and term frequency (corpus)
    # term frequency of term in corresponding doc
    # returns the max frequency of a term in whole doc corpus
    @staticmethod
    def get_max_freq_term(matrix):
        max_freq = 0
        for k, v in matrix.items():
            for word in v:
                if v[word] > max_freq:
                    max_freq = v[word]
        return max_freq

    # matrix: contains matrix of docs with terms and term frequency (corpus)
    # create idf vector
    @staticmethod
    def create_idf_matrix(matrix):
        idf_matrix = {}
        num_docs = len(matrix.items())
        for doc, terms in matrix.items():
            for term in terms:
                if term not in idf_matrix:
                    idf_matrix.update({term: 1})
                else:
                    count = idf_matrix[term]
                    idf_matrix.update({term: count+1})
        for word in idf_matrix:
            idf_matrix[word] = math.log(num_docs/idf_matrix[word], 10)
        return idf_matrix

    # Implementation like in slides Lecture 4 p.14
    # tf(t,d) = (1 + log10(ft,d)) / (1 + log10 (max{ft’,d : t’ ∈ d}))
    @staticmethod
    def create_tf_matrix(matrix):
        tf_matrix = {}
        max_freq = TFIDF.get_max_freq_term(matrix)
        for k, v in matrix.items():
            for word in v:
                counter = 1+math.log(v[word], 10)
                denominator = 1+math.log(max_freq, 10)
                value = counter/denominator
                Utils.add_ele_to_matrix(tf_matrix, k, word, value)
        return tf_matrix

    # create the tf-idf matrix
    def create_tf_idf_matrix(self):
        matrix = self.create_count_matrix()
        # create the idf vector and the tf matrices
        self.idf_vector = TFIDF.create_idf_matrix(matrix)
        tf_matrix = TFIDF.create_tf_matrix(matrix)
        tf_idf_matrix = {}
        for k, v in tf_matrix.items():
            for word in v:
                value = v[word] * self.idf_vector[word]
                Utils.add_ele_to_matrix(tf_idf_matrix, k, word, value)
        return tf_idf_matrix

    # creates a query vector
    # query: list of query strings
    def create_query_vector(self, query):
        tokens = query.split()
        tokens_set = set(tokens)
        word_dict = dict.fromkeys(tokens_set, 0)
        for word in tokens:
            word_dict[word] += 1
        for k, v in word_dict.items():
            try:
                word_dict[k] = self.idf_vector[k] * v
            except KeyError:
                # KeyError only when term is not in idf vector
                # thus, ignore the term by removing from query
                word_dict.pop(k)
        return word_dict
