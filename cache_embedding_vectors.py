# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'

import pickle
import numpy as np
import glob
import os
import os.path

from preprocessing import Preprocess
from utils import Utils
from tf_idf import TFIDF
from query_expansion.rocchio import RocchioOptimizeQuery
from performance import Performance


class Caching:
    """
    Prerequisites:
    if there are no process_data or cache directories or they are empty
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
            self.corpus = pickle.load(open('process_data/lemma_processed_paragraph.pkl', 'rb'))
            self.queries = pickle.load(open('process_data/lemma_processed_query.pkl', 'rb'))
            self.tf_idf_cache = 'cache/lemma_tf_idf.pkl'
        elif process_type == 'stem':
            self.corpus = pickle.load(open('process_data/processed_paragraph.pkl', 'rb'))
            self.queries = pickle.load(open('process_data/processed_query.pkl', 'rb'))
            self.tf_idf_cache = 'cache/tf_idf.pkl'

        self.cached_embeddings = pickle.load(open('cache/word_embedding_vectors.pkl', 'rb'))

        # dimension of the embedding vector
        # we are currently using the pre-trained glove model
        # glove.840.300d
        self.vector_dimension = vector_dimension

        # create queries
        self.raw_query = pickle.load(open('process_data/raw_query.pkl', 'rb'))
        # self.query_map = Caching.create_query_map(self.raw_query, self.process_type)
        self.test = pickle.load(open('process_data/simulated_test.pkl', 'rb'))
        # self.unique_doc = np.unique(list(self.test['docid']))
        self.paragraph_ids = pickle.load(open('process_data/paragraph_ids.pkl', 'rb'))
        self.doc_structure = self.create_document_corpus()

        # cache file paths
        self.avg_query_embeddings = 'cache/avg_query_embeddings.pkl'
        self.avg_doc_embeddings = 'cache/avg_doc_embeddings.pkl'
        self.avg_query_expanded_embeddings = 'cache/avg_query_expanded_embeddings.pkl'

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
        It takes lemma_process_paragraphs.pkl as input.
        :return: Returns a dict of {docid: {term: value, ...}, ...}
        """
        print('create document corpus as dict...')
        return Utils.get_document_structure_from_data(self.test, self.paragraph_ids, self.corpus)

    def create_query_embeddings(self, tf_idf):
        """
        create cache for query average word embedding vector
        :return:  Returns a dict of {processed query: embedding vector, ...}
        """
        query_embedding_vector = {}

        for q in self.queries:
            sum_embedding_vectors = np.zeros(self.vector_dimension)  # create initial empty array
            terms = q.split()

            query_vector = tf_idf.create_query_vector(q)

            sum_weight = 0

            for term in terms:
                try:
                    weight = query_vector[term]
                    sum_weight += weight
                    we = np.multiply(weight, self.cached_embeddings[term])
                    sum_embedding_vectors = np.add(sum_embedding_vectors, we)
                except KeyError:
                    pass

            for idx in range(self.vector_dimension):
                sum_embedding_vectors[idx] = sum_embedding_vectors[idx] / sum_weight

            query_embedding_vector.update({q: sum_embedding_vectors})
        # avg_query_embeddings.pkl contains {process_query: nparray, ...}
        pickle.dump(query_embedding_vector, open(self.avg_query_embeddings, 'wb'))

    def create_query_embeddings_query_expansion(self, tf_idf, top_k=10, rocchio_terms=5):
        """
        Prerequisites: requires cached bm25 scores in cache.
        You have to create the bm25_scores.pkl first!

        See: create_features.py and run calculate_bm25 function to create it.

        create cache for query average word embedding vector

        :param tf_idf: The TFIDF Object
        :param top_k: top_k documents for rocchio to consider
        :param rocchio_terms: number of terms rocchio should include
        :return:  Returns a dict of {processed query: embedding vector, ...}
        """
        p = Performance()
        query_embedding_vector = {}
        print('Load BM25 scores...')
        bm25_scores = pickle.load(open('cache/bm25_scores.pkl', 'rb'))

        for q in self.queries:
            sum_embedding_vectors = np.zeros(self.vector_dimension)  # create initial empty array
            tfidf_q = tf_idf.create_query_vector(q)
            rocchio = RocchioOptimizeQuery(query_vector=tfidf_q, tf_idf_matrix=tf_idf.term_doc_matrix)
            scores = {}
            try:
                scores = bm25_scores[q]
            except KeyError:
                pass

            relevant_docs = p.filter_relevance_by_top_k(scores, top_k)
            non_relevant_docs = p.filter_pred_negative(scores)
            new_query = rocchio.execute_rocchio(relevant_docs, non_relevant_docs, rocchio_terms)

            sum_weight = 0

            for term, value in new_query.items():
                try:
                    weight = value
                    sum_weight += weight
                    we = np.multiply(weight, self.cached_embeddings[term])
                    sum_embedding_vectors = np.add(sum_embedding_vectors, we)
                except KeyError:
                    pass

            for idx in range(self.vector_dimension):
                sum_embedding_vectors[idx] = sum_embedding_vectors[idx] / sum_weight

            query_embedding_vector.update({q: sum_embedding_vectors})
        # avg_query_embeddings.pkl contains {process_query: nparray, ...}
        pickle.dump(query_embedding_vector, open(self.avg_query_expanded_embeddings, 'wb'))
        return query_embedding_vector

    def create_document_embeddings(self, tf_idf):
        """
        create cache for document average word embedding vector
        :return: Returns a dict of {doc ids: embedding vectors, ...}
        """

        doc_embedding_vectors = {}
        for docid, terms in tf_idf.items():
            sum_weight = 0
            sum_embedding_vectors = np.zeros(self.vector_dimension)
            number_terms = len(terms.keys())
            if number_terms > 0:
                for term in terms.keys():
                    try:
                        weight = terms[term]
                        we = np.multiply(weight, self.cached_embeddings[term])
                        sum_embedding_vectors = np.add(sum_embedding_vectors, we)
                        sum_weight += weight
                    except KeyError:
                        pass
                if sum_weight > 0:
                    sum_embedding_vectors /= sum_weight
                else:
                    sum_embedding_vectors = np.zeros(self.vector_dimension)

            doc_embedding_vectors.update({docid: sum_embedding_vectors})
        # avg_doc_embeddings.pkl contains {docid: nparray, ...}
        pickle.dump(doc_embedding_vectors, open(self.avg_doc_embeddings, 'wb'))

    @staticmethod
    def clear_cache():
        """
        https://stackoverflow.com/questions/1995373/deleting-all-files-in-a-directory-with-python/1995397
        :return:
        """
        filelist = glob.glob(os.path.join('cache', "*.pkl"))
        for f in filelist:
            os.remove(f)
