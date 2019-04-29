# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'

import pickle
import numpy as np

from preprocessing import Preprocess
from utils import Utils


class Caching:
    """
    Prerequisites:
    if there are no processed_data or cache directories or they are empty
    - you ran preprocessing at least once
    - you ran cache_word_embeddings at least once
    """

    def __init__(self, vector_dimension=300, process_type='stem'):
        """

        :param vector_dimension: this attribute depends on the pre-trained glove file. Since we were using
        a 300d vectors file, we set this to 300 by default
        """
        self.process_type = process_type
        if process_type == 'lemma':
            self.corpus = pickle.load(open('processed_data/lemma_processed_paragraph.pkl', 'rb'))
            self.queries = pickle.load(open('processed_data/lemma_processed_query.pkl', 'rb'))
            self.tf_idf_cache = 'cache/lemma_tf_idf.pkl'
        elif process_type == 'stem':
            self.corpus = pickle.load(open('processed_data/processed_paragraph.pkl', 'rb'))
            self.queries = pickle.load(open('processed_data/processed_query.pkl', 'rb'))
            self.tf_idf_cache = 'cache/tf_idf.pkl'

        self.cached_embeddings = pickle.load(open('cache/word_embedding_vectors.pkl', 'rb'))

        # dimension of the embedding vector
        # we are currently using the pre-trained glove model
        # glove.840.300d
        self.vector_dimension = vector_dimension

        # create queries
        self.raw_query = pickle.load(open('processed_data/raw_query.pkl', 'rb'))
        # self.query_map = Caching.create_query_map(self.raw_query, self.process_type)
        self.test = pickle.load(open('processed_data/simulated_test.pkl', 'rb'))
        # self.unique_doc = np.unique(list(self.test['docid']))
        self.paragraph_ids = pickle.load(open('processed_data/paragraph_ids.pkl', 'rb'))
        self.doc_structure = self.create_document_corpus()

        # cache file paths
        self.avg_query_embeddings = 'cache/avg_query_embeddings.pkl'
        self.avg_doc_embeddings = 'cache/avg_doc_embeddings.pkl'

    @staticmethod
    def create_query_map(raw_query, process_type):
        """
        Creates
        :param process_type: can be either stem or lemma
        :param raw_query: The unprocessed list of queries
        :return: a dict of {unprocessed query: processed query, ...}
        """
        print('Create Query Map...')
        query_map = {}
        for idx, query in enumerate(raw_query):
                prep = Preprocess.preprocess(process_type, [query[7:].replace("/", " ").replace("%20", " ")])
                query_map.update({query: prep[0]})
        return query_map

    def create_document_corpus(self):
        """
        Creates the term-document count matrix.
        It takes lemma_processed_paragraphs.pkl as input.
        :return: Returns a dict of {docid: {term: value, ...}, ...}
        """
        print('create document corpus as dict...')
        return Utils.get_document_structure_from_data(self.test, self.paragraph_ids, self.corpus)

    def create_query_embeddings(self):
        """
        create cache for query average word embedding vector
        :return:  Returns a dict of {processed query: embedding vector, ...}
        """
        query_embedding_vector = {}
        for q in self.queries:
            sum_embedding_vectors = np.zeros(self.vector_dimension)  # create initial empty array
            number_terms = len(self.queries)  # length of query
            terms = q.split()

            for term in terms:
                try:
                    sum_embedding_vectors = np.add(sum_embedding_vectors, self.cached_embeddings[term])
                except KeyError:
                    # print('passed', term)
                    pass

            for idx in range(self.vector_dimension):
                sum_embedding_vectors[idx] = sum_embedding_vectors[idx] / number_terms

            query_embedding_vector.update({q: sum_embedding_vectors})
        # avg_query_embeddings.pkl contains {processed_query: nparray, ...}
        pickle.dump(query_embedding_vector, open(self.avg_query_embeddings, 'wb'))

    def create_document_embeddings(self):
        """
        create cache for document average word embedding vector
        :return: Returns a dict of {doc ids: embedding vectors, ...}
        """
        doc_embedding_vectors = {}
        for docid, terms in self.doc_structure.items():
            sum_embedding_vectors = np.zeros(self.vector_dimension)
            number_terms = len(terms.keys())
            if number_terms > 0:
                for term in terms.keys():
                    try:
                        sum_embedding_vectors = np.add(sum_embedding_vectors, self.cached_embeddings[term])
                    except KeyError:
                        # print('passed', term)
                        pass
                sum_embedding_vectors /= number_terms

            doc_embedding_vectors.update({docid: sum_embedding_vectors})
        # avg_doc_embeddings.pkl contains {docid: nparray, ...}
        pickle.dump(doc_embedding_vectors, open(self.avg_doc_embeddings, 'wb'))











