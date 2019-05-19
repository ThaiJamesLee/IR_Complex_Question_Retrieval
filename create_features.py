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

    def generate_bm25_doc_doc(self, file=None):
        """
        Generates all features for doc: doc: score.
        This includes tf-idf, tf-idf + rocchio, bm25, glove, glove + rocchio
        :param file specify a file name
        :return:
        """
        # where to store every result


        bm25 = BM25(self.caching.doc_structure)

        # contains docid: list(terms)
        docs = self.caching.create_doc_terms()
        print('Calculate BM25...')
        if file is None:
            filepath_bm25 = f'{self.folder}doc_doc_bm25_scores.pkl'
        else:
            filepath_bm25 = file

        scores = dict()
        counter = 1
        for (doc, terms) in docs:
            print(doc, f' {counter} / {len(docs)}')
            counter += 1

            # score contains dict of docid: score
            score = bm25.compute_relevance_on_corpus_list(terms)
            scores.update({doc: score})
        print(f'Store BM25 scores in {filepath_bm25}')
        pickle.dump(scores, open(filepath_bm25, 'wb'))
        print('Saved.')

    def generate_cosine_tfidf_doc_doc(self, file=None):
        """
        Calculate cosine similarities of tf idf, where one document is a query against all other
        :return:
        """
        print('Calculate cosine similarites TF-IDF...')
        if file is None:
            filepath = f'{self.folder}doc_doc_tfidf_scores.pkl'
        else:
            filepath = file

        tf_idf = self.tf_idf
        scores = {}

        counter = 1
        num_q = len(tf_idf.term_doc_matrix.keys())

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

    def generate_cosine_tfidf_rocchio_doc_doc(self, top_k=10, rocchio_terms=5, file=None):
        """

        :param top_k: setting for rocchio to consider which bm25 the number of most similar docs
        :param rocchio_terms: number of terms to add to the new query (in this case document
        :return:
        """
        print('Calculate cosine similarites TF-IDF + Rocchio...')

        bm25_scores = pickle.load(open(f'{self.folder}doc_doc_bm25_scores.pkl', 'rb'))
        p = Performance()

        filepath = f'{self.folder}doc_doc_tfidf_rocchio_scores_{rocchio_terms}.pkl'
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

    def generate_cosine_glove_doc_doc(self, file=None):
        """

        :return:
        """
        print('Calculate cosine similarites Glove...')
        doc_glove_vectors = pickle.load(open(self.caching.avg_doc_embeddings, 'rb'))
        filepath = f'{self.folder}doc_doc_glove_scores.pkl'

        counter = 1
        num_q = len(doc_glove_vectors.keys())

        scores = {}

        try:
            open(filepath, 'rb')
            print(f'Scores already cached in {filepath}')
        except FileNotFoundError:
            for doc, vec in doc_glove_vectors.items():

                print(doc, f'{counter} / {num_q}')
                counter += 1

                similarity = {}
                for doc1, vec1 in doc_glove_vectors.items():
                    score = Similarity.cosine_similarity_array(vec, vec1)
                    if score > 0:
                        similarity.update({doc1: score})
                scores.update({doc: similarity})

            print(f'Store Glove scores in {filepath}')
            pickle.dump(scores, open(filepath, 'wb'))
            print('Saved.')

    def generate_cosine_glove_rocchio_doc_doc(self, top_k=10, rocchio_terms=5, file=None):
        """

        :param top_k: setting for rocchio to consider which bm25 the number of most similar docs
        :param rocchio_terms: number of terms to add to the new query (in this case document
        :return:
        """
        print('Calculate cosine similarites Glove + Rocchio...')

        # Check if glove rocchio document vectors already exists
        # if not then calculate them first
        glove_rocchio_file = f'{self.folder}docs_glove_rocchio_{rocchio_terms}.pkl'
        try:
            glove_rocchio_docs = pickle.load(open(glove_rocchio_file, 'rb'))
            print(f'Loaded cached expanded doc embedding vectors {glove_rocchio_file}')
        except FileNotFoundError:
            # documents with query expansion not yet there
            # thus, we create them here
            tf_idf = self.tf_idf
            print('Create document word embeddings with rocchio...')
            self.caching.create_document_embeddings_rocchio(tf_idf, glove_rocchio_file, top_k=top_k, rocchio_terms=rocchio_terms)
            glove_rocchio_docs = pickle.load(open(glove_rocchio_file, 'rb'))

        doc_glove_vectors = pickle.load(open(self.caching.avg_doc_embeddings, 'rb'))
        filepath = f'{self.folder}doc_doc_glove_rocchio_scores_{rocchio_terms}.pkl'

        counter = 1
        num_q = len(glove_rocchio_docs.keys())

        scores = {}

        try:
            open(filepath, 'rb')
            print(f'Scores already cached in {filepath}')
        except FileNotFoundError:
            for doc, vec in glove_rocchio_docs.items():

                print(doc, f'{counter} / {num_q}')
                counter += 1

                similarity = {}
                for doc1, vec1 in doc_glove_vectors.items():
                    score = Similarity.cosine_similarity_array(vec, vec1)
                    if score > 0:
                        similarity.update({doc1: score})
                scores.update({doc: similarity})

            print(f'Store Glove + Rocchio scores in {filepath}')
            pickle.dump(scores, open(filepath, 'wb'))
            print('Saved.')

    def generate_scores_doc_doc(self, rocchio_terms=5, top_k=10):
        """
        NOT TESTED!!!
        :param rocchio_terms: number of terms to add to new query if using rocchio
        :param top_k: number of most relevant document for rocchio to consider
        :return:
        """
        filepath = f'{self.folder}doc_doc_scores.pkl'

        # Check if glove rocchio document vectors already exists
        # if not then calculate them first
        glove_rocchio_file = f'{self.folder}docs_glove_rocchio_{rocchio_terms}.pkl'

        # initialize for bm25
        bm25 = BM25(self.caching.doc_structure)
        docs = self.caching.create_doc_terms()

        # initialize for tf idf
        tf_idf = self.tf_idf

        try:
            glove_rocchio_docs = pickle.load(open(glove_rocchio_file, 'rb'))
            print(f'Loaded cached expanded doc embedding vectors {glove_rocchio_file}')
        except FileNotFoundError:
            # documents with query expansion not yet there
            # thus, we create them here
            print('Create document word embeddings with rocchio...')
            self.caching.create_document_embeddings_rocchio(tf_idf, glove_rocchio_file, top_k=top_k,
                                                            rocchio_terms=rocchio_terms)
            glove_rocchio_docs = pickle.load(open(glove_rocchio_file, 'rb'))

        doc_glove_vectors = pickle.load(open(self.caching.avg_doc_embeddings, 'rb'))

        try:
            open(filepath, 'rb')
            print(f'Scores already cached in {filepath}')
        except FileNotFoundError:

            print('Calculate bm25 scores...')
            scores = dict()
            counter = 1
            for (doc, terms) in docs:
                print(doc, f' {counter} / {len(docs)}')
                counter += 1

                # score contains dict of docid: score
                score = bm25.compute_relevance_on_corpus_list(terms)
                scores.update({doc: score})

            print('Calculate TF-IDF scores...')
            counter = 1
            num_q = len(tf_idf.term_doc_matrix.keys())

            for k, v in tf_idf.term_doc_matrix.items():

                print(k, f'{counter} / {num_q}')
                counter += 1
                for k1, v1 in tf_idf.term_doc_matrix.items():
                    score = Similarity.cosine_similarity_normalized(v, v1)
                    Utils.append_ele_to_matrix_list(scores, k, k1, [0, score, 0, 0, 0])

            print('Calculate TF-IDF + Rocchio scores...')
            try:
                bm25_scores = pickle.load(open(f'{self.folder}doc_doc_bm25_scores.pkl', 'rb'))
            except FileNotFoundError:
                print('BM25 scores not available yet. Calculate the relevance scores')
                self.generate_bm25_doc_doc()
                bm25_scores = pickle.load(open(f'{self.folder}doc_doc_bm25_scores.pkl', 'rb'))

            p = Performance()
            counter = 1
            num_q = len(tf_idf.term_doc_matrix.keys())

            for k, v in tf_idf.term_doc_matrix.items():

                print(k, f'{counter} / {num_q}')
                counter += 1

                rocchio = RocchioOptimizeQuery(query_vector=v, tf_idf_matrix=tf_idf.term_doc_matrix)
                relevant_docs = p.filter_relevance_by_top_k(bm25_scores[k], top_k)
                non_relevant_docs = p.filter_pred_negative(bm25_scores[k])
                new_query = rocchio.execute_rocchio(relevant_docs, non_relevant_docs, rocchio_terms)

                for k1, v1 in tf_idf.term_doc_matrix.items():
                    score = Similarity.cosine_similarity_normalized(new_query, v1)
                    Utils.append_ele_to_matrix_list(scores, k, k1, [0, 0, score, 0, 0])

            print('Calculate glove scores...')
            counter = 1
            num_q = len(doc_glove_vectors.keys())

            for doc, vec in doc_glove_vectors.items():

                print(doc, f'{counter} / {num_q}')
                counter += 1

                for doc1, vec1 in doc_glove_vectors.items():
                    score = Similarity.cosine_similarity_array(vec, vec1)
                    Utils.append_ele_to_matrix_list(scores, doc, doc1, [0, 0, 0, score, 0])

            print('Calculate glove + rocchio scores...')
            counter = 1
            num_q = len(glove_rocchio_docs.keys())

            for doc, vec in glove_rocchio_docs.items():

                print(doc, f'{counter} / {num_q}')
                counter += 1

                for doc1, vec1 in doc_glove_vectors.items():
                    score = Similarity.cosine_similarity_array(vec, vec1)
                    Utils.append_ele_to_matrix_list(scores, doc, doc1, [0, 0, 0, 0, score])

            # store scores
            print(f'Store scores in {filepath}')
            pickle.dump(scores, open(filepath, 'wb'))
            print('Saved.')


# print('================== Load Data ===================')

"""
feature_generator = FeatureGenerator()
feature_generator.generate_bm25_doc_doc()

feature_generator.generate_cosine_tfidf_doc_doc()
feature_generator.generate_cosine_tfidf_rocchio_doc_doc()

feature_generator.generate_cosine_glove_doc_doc()
feature_generator.generate_cosine_glove_rocchio_doc_doc()
"""
# feature_generator = FeatureGenerator()
# feature_generator.generate_cosine_glove_rocchio_doc_doc()
# feature_generator.generate_cosine_tfidf_doc_doc()
# feature_generator.generate_cosine_tfidf_rocchio_doc_doc()
# feature_generator.generate_cosine_glove_doc_doc()
# feature_generator.generate_scores_doc_doc()

# feature_generator.calculate_cosine_semantic_embeddings_query_expansion()
# feature_generator.calculate_cosine_semantic_embeddings()

# feature_generator.create_cache()
# feature_generator.calculate_cosine_semantic_embeddings()

# feature_generator.generate_features_scores_and_cache()




