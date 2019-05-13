# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'

import pickle
import pandas as pd
import numpy as np

from bm25 import BM25
from tf_idf import TFIDF

from query_expansion.rocchio import RocchioOptimizeQuery
from performance import Performance
from similarity import Similarity
from cache_embedding_vectors import Caching
from utils import Utils


class FeatureGenerator:
    """
    Prerequisites:
    if process_data and cache directory do not exists or are empty
    - you run pre-process at least once
    - the cached word embeddings must exists. Thus, you need to run 'cache_word_embeddings.py'
    and 'cache_embedding_vectors.py' first, if not already.

    The create_cache function might be necessary if cache folder is empty.
    Only the generate_feature function needs to be called if requirements are met.
    """

    def __init__(self, caching=None, tf_idf=None):
        """

        :param caching: a Cache instance
        :param tf_idf: a TFIDF insance
        """

        # set of predefined strings containing some file paths
        # change here the files to load
        self.paragraph_corpus_file = 'process_data/lemma_process_paragraph.pkl'
        self.paragraph_id_file = 'process_data/paragraph_ids.pkl'

        # set here where to save the cached data
        self.bm25_scores_file = 'cache/bm25_scores.pkl'
        self.similarity_tf_idf_scores_file = 'cache/cosine_tf_idf.pkl'
        self.similarity_if_idf_scores_query_exp_file = 'cache/cosine_tf_idf_qe.pkl'
        self.similarity_semantic_word_embedding_scores_file = 'cache/cosine_sem_we.pkl'
        self.similarity_query_expansion_file = 'cache/cosine_query_expansion_'
        self.features_dataframe_file = 'cache/features_dataframe.pkl'
        self.cosine_glove = 'cache/cosine_sem_we.pkl'
        self.cosine_glove_we = 'cache/cosine_sem_we_query_exp.pkl'

        # where to save cached data for doc-doc pairs
        self.folder = 'documents_retrieval/'

        if caching is None:
            self.caching = Caching(process_type='lemma')
        else:
            self.caching = caching

        if tf_idf is None:
            self.tf_idf = TFIDF(self.caching.doc_structure)
        else:
            self.tf_idf = tf_idf
        self.num_queries = len(self.caching.queries)

        # a cache use for various stuff
        self.temp_cache = set()

    def calculate_bm25(self):
        """"
        Calculate the relevance scores with BM25, and cache them in a file.
        """
        try:
            open(self.bm25_scores_file, 'rb')
            print('Scores already in cache.')
        except FileNotFoundError:
            bm25 = BM25(self.caching.doc_structure, k=1.2)

            # save the relevance scores as dict
            rel_scores = {}
            counter = 1
            for q in self.caching.queries:
                print(f'{q}: {counter} / {self.num_queries}')
                counter += 1

                # contains dict of docid: relevance score
                scores = bm25.compute_relevance_on_corpus(q)
                # scores = Performance.filter_relevance_by_threshold(scores)
                rel_scores.update({q: scores})

            # dump in specified file
            print('Save relevance scores...')
            pickle.dump(rel_scores, open(self.bm25_scores_file, 'wb'))
            print('Saved in', self.bm25_scores_file)

    def calculate_cosine_tf_idf(self):
        """
        Calculate cosine similarities of queries and documents, and cache them in a file.
        """
        print('Calculate cosine for tf-idf...')
        # try to open cached file
        # if not exist, then
        try:
            open(self.similarity_tf_idf_scores_file, 'rb')
            print('Scores already in cache.')
        except FileNotFoundError:
            tf_idf = self.tf_idf

            counter = 1
            similarity_scores_tf_idf = {}
            for q in self.caching.queries:

                print(f'{q}: {counter} / {self.num_queries}')
                counter += 1

                query_vector = tf_idf.create_query_vector(q)

                similarities = {}

                for docid, terms in tf_idf.term_doc_matrix.items():
                    score = Similarity.cosine_similarity_normalized(query_vector, terms)
                    if score > 0:
                        similarities.update({docid: score})

                similarity_scores_tf_idf.update({q: similarities})

            # dump in specified file
            print('Save similarity scores...')
            # contains {query: {docid: cosine, ...}, ...}
            pickle.dump(similarity_scores_tf_idf, open(self.similarity_tf_idf_scores_file, 'wb'))
            print('Saved in', self.similarity_tf_idf_scores_file)

    def calculate_cosine_tf_idf_rocchio(self, top_k=10, rocchio_terms=5):
        print('Calculate similarity with query expansion')
        p = Performance()

        bm25_relevance_scores = pickle.load(open(self.bm25_scores_file, 'rb'))

        tf_idf = self.tf_idf

        similarity_scores_tf_idf = {}
        counter = 1
        for q in self.caching.queries:
            query_vector = tf_idf.create_query_vector(q)

            print(f'{q}: {counter} / {self.num_queries}')
            counter += 1

            rocchio = RocchioOptimizeQuery(query_vector=query_vector, tf_idf_matrix=tf_idf.term_doc_matrix)

            relevant_docs = p.filter_relevance_by_top_k(bm25_relevance_scores[q], top_k)
            non_relevant_docs = p.filter_pred_negative(bm25_relevance_scores[q])
            new_query = rocchio.execute_rocchio(relevant_docs, non_relevant_docs, rocchio_terms)

            similarities = {}
            for docid, terms in tf_idf.term_doc_matrix.items():
                score = Similarity.cosine_similarity_normalized(new_query, terms)
                if score > 0:
                    similarities.update({docid: score})
            similarity_scores_tf_idf.update({q: similarities})

        print('Save similarities and expanded queries...')
        pickle.dump(similarity_scores_tf_idf, open(self.similarity_query_expansion_file + f'{rocchio_terms}.pkl', 'wb'))
        print('Saved.')

    def calculate_cosine_glove(self):

        # cosine for avg embedding vector
        print('Load cached embeddings...')

        query_embeddings = pickle.load(open(self.caching.avg_query_embeddings, 'rb'))
        document_embeddings = pickle.load(open(self.caching.avg_doc_embeddings, 'rb'))

        num_queries = len(query_embeddings.keys())

        print('Calculate cosine for avg embedding vectors...')
        counter = 1
        similarity_scores_we = {}
        for query, vector in query_embeddings.items():

            print(f'{query}: {counter} / {num_queries}')
            counter += 1

            similarities = {}
            for doc, doc_vec in document_embeddings.items():
                score = Similarity.cosine_similarity_array(vector, doc_vec)
                if score > 0:
                    similarities.update({doc: score})
            similarity_scores_we.update({query: similarities})

        # dump similarity scores in {query: {docid: score, ...}, ...}
        print('Dump scores in ', self.cosine_glove)
        pickle.dump(similarity_scores_we, open(self.cosine_glove, 'wb'))
        # print(similarity_scores_we)

    def calculate_cosine_glove_and_rocchio(self, rocchio_terms=5):

        # cosine for avg embedding vector
        print('Load cached embeddings...')

        query_embeddings = pickle.load(open(self.caching.avg_query_expanded_embeddings + f'{rocchio_terms}.pkl', 'rb'))
        document_embeddings = pickle.load(open(self.caching.avg_doc_embeddings, 'rb'))

        print('Calculate cosine for avg embedding vectors...')

        counter = 1
        similarity_scores_we = {}

        num_queries = len(query_embeddings.keys())

        for query, vector in query_embeddings.items():

            print(f'{query}: {counter} / {num_queries}')
            counter += 1

            similarities = {}
            for doc, doc_vec in document_embeddings.items():
                score = Similarity.cosine_similarity_array(vector, doc_vec)
                if score > 0:
                    similarities.update({doc: score})

            similarity_scores_we.update({query: similarities})

        # dump similarity scores in {query: {docid: score, ...}, ...}
        print('Dump scores in ', self.cosine_glove_we)
        pickle.dump(similarity_scores_we, open(self.cosine_glove_we, 'wb'))

    def create_cache(self):
        """
        Not necessary if cache folder already contains the files
        1. avg_doc_embeddings.pkl
        2. avg_query_embeddings.pkl
        """
        print('Create query embeddings...')
        self.caching.create_query_embeddings()
        print('saved.')

        print('Create document embeddings...')
        self.caching.create_document_embeddings()
        print('saved.')

    def generate_bm25_doc_doc(self):
        """
        Generates all features for doc: doc: score.
        This includes tf-idf, tf-idf + rocchio, bm25, glove, glove + rocchio
        :return:
        """
        # where to store every result

        docids = self.caching.doc_structure.keys()

        bm25 = BM25(self.caching.doc_structure)

        # contains docid: list(terms)
        docs = self.caching.create_doc_terms()
        print('Calculate BM25...')
        filepath_bm25 = f'{self.folder}doc_doc_bm25_scores.pkl'
        try:
            open(filepath_bm25, 'rb')
            print(f'Scores already cached in {filepath_bm25}')
        except FileNotFoundError:
            scores = dict()
            counter = 1
            for (doc, terms) in docs:
                print(doc, f' {counter} / {len(docs)}')
                counter += 1

                score = bm25.compute_relevance_on_corpus(terms)
                scores.update({doc: score})
            print(scores)
            print(f'Store BM25 scores in {filepath_bm25}')
            pickle.dump(scores, open(filepath_bm25, 'wb'))
            print('Saved.')

    def generate_cosine_tfidf_doc_doc(self):
        """
        Calculate cosine similarities of tf idf, where one document is a query against all other
        :return:
        """
        filepath = f'{self.folder}doc_doc_tfidf_scores.pkl'
        tf_idf = self.tf_idf

        counter = 1
        num_q = len(tf_idf.term_doc_matrix.keys())

        scores = {}
        try:
            open(filepath, 'rb')
            print(f'Scores already cached in {filepath}')
        except FileNotFoundError:
            for k, v in tf_idf.term_doc_matrix.items():

                print(k, f'{counter} / {num_q}')
                counter += 1
                similarity = {}
                for k1, v1 in tf_idf.term_doc_matrix.items():
                    score = Similarity.cosine_similarity_normalized(v, v1)
                    if score > 0:
                        similarity.update({k1: score})
                scores.update({k: similarity})

            print(f'Store TF-IDF scores in {filepath}')
            pickle.dump(scores, open(filepath, 'wb'))
            print('Saved.')

    def generate_cosine_tfidf_rocchio_doc_doc(self, top_k=10, rocchio_terms=5):
        """

        :param top_k: setting for rocchio to consider which bm25 the number of most similar docs
        :param rocchio_terms: number of terms to add to the new query (in this case document
        :return:
        """

        bm25_scores = pickle.load(open(f'{self.folder}doc_doc_bm25_scores.pkl', 'rb'))
        p = Performance()

        filepath = f'{self.folder}doc_doc_tfidf_rocchio_scores.pkl'
        tf_idf = self.tf_idf

        counter = 1
        num_q = len(tf_idf.term_doc_matrix.keys())

        scores = {}
        try:
            open(filepath, 'rb')
            print(f'Scores already cached in {filepath}')
        except FileNotFoundError:
            for k, v in tf_idf.term_doc_matrix.items():

                print(k, f'{counter} / {num_q}')
                counter += 1

                rocchio = RocchioOptimizeQuery(query_vector=v, tf_idf_matrix=tf_idf.term_doc_matrix)
                relevant_docs = p.filter_relevance_by_top_k(bm25_scores[k], top_k)
                non_relevant_docs = p.filter_pred_negative(bm25_scores[k])
                new_query = rocchio.execute_rocchio(relevant_docs, non_relevant_docs, rocchio_terms)

                similarity = {}
                for k1, v1 in tf_idf.term_doc_matrix.items():
                    score = Similarity.cosine_similarity_normalized(new_query, v1)
                    if score > 0:
                        similarity.update({k1: score})
                scores.update({k: similarity})

            print(f'Store TF-IDF + Rocchio scores in {filepath}')
            pickle.dump(scores, open(filepath, 'wb'))
            print('Saved.')

    def generate_cosine_glove_doc_doc(self):

        doc_glove_vectors = pickle.load(self.caching.avg_doc_embeddings)
        filepath = f'{self.folder}doc_doc_glove_scores.pkl'


        counter = 1
        num_q = len(doc_glove_vectors.keys())

        scores = {}

        for doc, vec in doc_glove_vectors.items():

            print(doc, f'{counter} / {num_q}')
            counter += 1

            similarity = {}
            for doc1, vec1 in doc_glove_vectors.items():
                score = Similarity.cosine_similarity_normalized(vec, vec1)
                if score > 0:
                    similarity.update({doc1: score})
            scores.update({doc: similarity})

        print(f'Store Glove scores in {filepath}')
        pickle.dump(scores, open(filepath, 'wb'))
        print('Saved.')

    def generate_cosine_glove_rocchio_doc_doc(self, top_k=10, rocchio_terms=5):
        glove_rocchio_file = f'{self.folder}doc_doc_glove_rocchio.pkl'
        try:
            glove_rocchio_docs = open(glove_rocchio_file, 'rb')
        except FileNotFoundError:
            # documents with query expansion not yet there
            # thus, we create them here
            tf_idf = self.tf_idf
            print('Create document word embeddings with rocchio...')
            glove_rocchio_docs = self.caching.create_document_embeddings_rocchio(tf_idf.term_doc_matrix, glove_rocchio_file, top_k=top_k, rocchio_terms=rocchio_terms)

        doc_glove_vectors = pickle.load(self.caching.avg_doc_embeddings)
        filepath = f'{self.folder}doc_doc_glove_scores.pkl'

        counter = 1
        num_q = len(glove_rocchio_docs.keys())

        scores = {}

        try:
            open(filepath, 'rb')
        except FileNotFoundError:
            for doc, vec in glove_rocchio_docs.items():

                print(doc, f'{counter} / {num_q}')
                counter += 1

                similarity = {}
                for doc1, vec1 in doc_glove_vectors.items():
                    score = Similarity.cosine_similarity_normalized(vec, vec1)
                    if score > 0:
                        similarity.update({doc1: score})
                scores.update({doc: similarity})

            print(f'Store Glove + Rocchio scores in {filepath}')
            pickle.dump(scores, open(filepath, 'wb'))
            print('Saved.')


# print('================== Load Data ===================')

feature_generator = FeatureGenerator()
# feature_generator.generate_bm25_doc_doc()

feature_generator.generate_cosine_tfidf_doc_doc()
feature_generator.generate_cosine_tfidf_rocchio_doc_doc()

feature_generator.generate_cosine_glove_doc_doc()
feature_generator.generate_cosine_glove_rocchio_doc_doc()
# feature_generator.calculate_cosine_semantic_embeddings_query_expansion()
# feature_generator.calculate_cosine_semantic_embeddings()

# feature_generator.create_cache()
# feature_generator.calculate_cosine_semantic_embeddings()

# feature_generator.generate_features_scores_and_cache()




