# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'


class Utils:
    """
    This class provide some operations on matrices.
    """
    @staticmethod
    def add_ele_to_matrix(matrix, doc, key, value):
        """

        :param matrix: The parent dict consist of docs as keys
        :param doc: The document id.
        :param key: The term that should be added.
        :param value: The value to the corresponding key (some numeric?)
        :return: Returns matrix with new element added.
        """
        if doc not in matrix:
            matrix.update({doc: {key: value}})
        else:
            matrix[doc].update({key: value})
        return matrix

    @staticmethod
    def create_vocabulary(corpus):
        """

        :param corpus: list of strings, each string consist of terms separated by whitespace.
        :return: set of terms.
        """
        vocabulary = set()
        for para in corpus:
            terms = para.split()
            words_set = set(terms)
            vocabulary = vocabulary.union(words_set)
        return vocabulary

    @staticmethod
    def create_vocabulary_from_dict(corpus):
        """

        :param corpus: {doc_id: {term: value, term: value, ...}, ...}
        :return: set of terms
        """
        vocabulary = set()
        for doc_id, terms in corpus.items():
            words_set = set(terms)
            vocabulary = vocabulary.union(words_set)
        return vocabulary

    @staticmethod
    def create_count_matrix(documents):
        """

        :param documents: List of strings, where terms separated by space.
        :return: Create from documents a term-doc matrix with number of occurrences of terms in corresponding doc
        """
        matrix = {}
        doc_id = 0
        for doc in documents:
            words = doc.split()
            words_set = set(words)
            word_dict = dict.fromkeys(words_set, 0)
            for word in words:
                word_dict[word] += 1
            doc = {str(doc_id): word_dict}
            matrix.update(doc)
            doc_id += 1
        return matrix

    @staticmethod
    def get_doc_and_relevance_by_query(query, df):

        """

        :param query: The unprocessed query.
        :param df: The dataframe that contains columns query, q0, docid, rel.
        :return:
        """
        y_true = df[df['query'] == query][['docid', 'rel']]
        return y_true


class Helper:
    """
    This class provides some helper functions, that makes life easier.
    """
    @staticmethod
    def print_short(text, size=200):
        """

        :param text: The String that should output, and cut if too long (text length > size parameter)
        :param size: Determines the maximum length of String until it gets cut. Default = 150
        :return: Returns String cut to size length if bigger than defined size.
        """
        text = (text[:size] + '...') if len(text) > size else text
        return text
