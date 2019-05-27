# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'
import numpy as np


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
        try:
            matrix[doc].update({key: value})
        except KeyError:
            matrix.update({doc: {key: value}})
        return matrix

    @staticmethod
    def append_ele_to_dict_list(dictionary, key, value):
        """
        Adds a k,v pair to a dict, where value is a list of values
        :param dictionary: dict containing {key: [values], ...}
        :param key: the key
        :param value: the value you want to append, must be of type str or list
        :return: returns the dictionary with the new key, value, where the value is append to a values list
        """
        try:
            dictionary = Utils.add_str_or_list(dictionary, key, value)
        except KeyError:
            dictionary = Utils.update_str_or_list(dictionary, key, value)
        return dictionary

    @staticmethod
    def add_str_or_list(dictionary, key, value):
        if type(value) == str or type(value) == int or type(value) == float or type(value) == np.float64:
            dictionary[key] = np.add(dictionary[key], value)
        elif type(value) == list:
            dictionary[key] = np.add(dictionary[key], value)
        else:
            raise Exception('Invalid type: the value must be of type str or list!')
        return dictionary

    @staticmethod
    def update_str_or_list(dictionary, key, value):
        if type(value) == str or type(value) == int or type(value) == float or type(value) == np.float64:
            dictionary.update({key: value})
        elif type(value) == list:
            dictionary.update({key: value})
        else:
            raise Exception('Invalid type: the value must be of type str or list!')
        return dictionary

    @staticmethod
    def append_ele_to_matrix_list(matrix, key1, key2, value):
        """
        Adds a k, k, v tuple to a dict that is called the matrix variable
        :param matrix: is a dictionary of the structure {docid: {docid: [values, ...]}, ...}
        :param key1: the key of the query or document id that is the query
        :param key2: the key of the document ranked against the given query or document
        :param value: the value you want to append, must be of type str or list
        :return: returns the matrix / dictionary containing the new key, key, value tuple
        """
        try:
            matrix = Utils.append_ele_to_dict_list(matrix[key1], key2, value)
        except KeyError:
            # if we get an key error, then the entry with the key pairs are empty and we need to create a new entry
            matrix.update({key1: Utils.update_str_or_list(dict(), key2, value)})
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
            print(len(vocabulary))
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
        :return: A datagrame with docid, rel as columns
        """
        y_true = df[df['query'] == query][['docid', 'rel']]
        return y_true

    @staticmethod
    def get_document_structure_from_data(df, corpus_ids, corpus):
        """ get term count structure for every doc in dataframe
        :param df: dataframe contains docid column
        :param corpus_ids: all docid in corpus
        :param corpus: all doc in corpus
        :return: document structure dict{docid, {term, count}}
        """
        unique_docid = set(df['docid'])
        unique_docs = [(docid, corpus[corpus_ids.index(docid)].split()) for docid in unique_docid]
        doc_structure = {}
        for (docid, doc) in unique_docs:
            value = {word: doc.count(word) for word in doc}
            doc_structure.update({docid: value})
        return doc_structure

    @staticmethod
    def get_document_term_from_data(df, corpus_ids, corpus):
        """

        :param df: a dataframe containing docids
        :param corpus_ids:
        :param corpus:
        :return:
        """
        unique_docid = set(df['docid'])
        unique_docs = [(docid, corpus[corpus_ids.index(docid)].split()) for docid in unique_docid]
        return unique_docs

    @staticmethod
    def get_document_term_from_data_as_string(df, corpus_ids, corpus):
        unique_docid = set(df['docid'])
        unique_docs = dict()
        for docid in unique_docid:
            unique_docs.update({docid: corpus[corpus_ids.index(docid)]})
        return unique_docs

    @staticmethod
    def get_value_from_key_in_dict(thedict, docid, key):
        """
        :param thedict: the dictionary, that should have numerics as values.
        :param key: The key of interest
        :return: Returns the numeric value if key exists, else 0.
        """
        try:
            return thedict[key][docid]
        except KeyError:
            return 0


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
