# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'

import pickle

from bm25 import BM25
from tf_idf import TFIDF

from query_expansion.rocchio import RocchioOptimizeQuery
from performance import Performance
from similarity import Similarity
from cache_query_and_docs_as_embedding_vectors import Caching


class FeatureGenerator:
    """
    Prerequisites:
    if processed_data and cache directory do not exists or are empty
    - you run pre-process at least once
    - the cached word embeddings must exists. Thus, you need to run 'cache_word_embeddings.py'
    and 'cache_query_and_docs_as_embedding_vectors.py' first, if not already.

    The create_cache function might be necessary if cache folder is empty.
    Only the generate_feature function needs to be called if requirements are met.
    """

    def __init__(self):

        # set of predefined strings containing some file paths
        # change here the files to load
        self.paragraph_corpus_file = 'processed_data/lemma_processed_paragraph.pkl'
        self.paragraph_id_file = 'processed_data/paragraph_ids.pkl'
        self.test_data_file = 'processed_data/simulated_test.pkl'

        self.avg_query_embeddings_file = 'cache/avg_query_embeddings.pkl'
        self.avg_doc_embeddings_file = 'cache/avg_doc_embeddings.pkl'

        # set here where to save the cached data
        self.bm25_scores_file = 'cache/bm25_scores.pkl'
        self.similarity_tf_idf_scores_file = 'cache/cosine_tf_idf.pkl'
        self.similarity_if_idf_scores_query_exp_file = 'cache/cosine_tf_idf_qe.pkl'
        self.similarity_semantic_word_embedding_scores_file = 'cache/cosine_sem_word_embeddings.pkl'

        self.caching = Caching(process_type='lemma')

        self.tf_idf = TFIDF(self.caching.doc_structure)

    def calculate_bm25(self):
        """"
        Calculate the relevance scores with BM25, and cache them in a file.
        """

        bm25 = BM25(self.caching.doc_structure, k=1.2)

        # save the relevance scores as dict
        rel_scores = {}

        for q in self.caching.queries:
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
        tf_idf = self.tf_idf

        similarity_scores_tf_idf = {}
        for q in self.caching.queries:
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

    def calculate_cosine_tf_idf_query_expansion(self, top_k=10):
        print('Calculate similarity with query expansion')
        p = Performance()
        bm25_relevance_scores = pickle.load(open(self.bm25_scores_file, 'rb'))
        tf_idf = self.tf_idf

        similarity_scores_tf_idf = {}
        query_expansion_cache = {}
        for q in self.caching.queries:
            query_vector = tf_idf.create_query_vector(q)

            rocchio = RocchioOptimizeQuery(query_vector=query_vector, tf_idf_matrix=tf_idf.term_doc_matrix)

            relevant_docs = p.filter_relevance_by_top_k(bm25_relevance_scores[q], top_k)
            non_relevant_docs = p.filter_pred_negative(bm25_relevance_scores[q])
            new_query = rocchio.execute_rocchio(relevant_docs, non_relevant_docs)

            query_expansion_cache.update({q: new_query})

            similarities = {}
            for docid, terms in tf_idf.term_doc_matrix.items():
                score = Similarity.cosine_similarity_normalized(new_query, terms)
                if score > 0:
                    similarities.update({docid: score})
            similarity_scores_tf_idf.update({q: similarities})

        print('Save similarities and expanded queries...')
        pickle.dump(similarity_scores_tf_idf, open('cache/cosine_query_expansion.pkl', 'wb'))
        pickle.dump(similarity_scores_tf_idf, open('cache/rocchio_expanded_queries.pkl', 'wb'))
        print('Saved.')

    def calculate_cosine_semantic_embeddings(self):

        # cosine for avg embedding vector
        print('Load cached embeddings...')

        query_embeddings = pickle.load(open(self.avg_query_embeddings_file, 'rb'))
        document_embeddings = pickle.load(open(self.avg_doc_embeddings_file, 'rb'))

        print('Calculate cosine for avg embedding vectors...')

        similarity_scores_we = {}
        for query, vector in query_embeddings.items():
            print('Calculate similarities for query vector = ', query)
            similarities = {}
            for doc, doc_vec in document_embeddings.items():
                score = Similarity.cosine_similarity_array(vector, doc_vec)
                if score > 0:
                    similarities.update({doc: score})
            similarity_scores_we.update({query: similarities})

        # dump similarity scores in {query: {docid: score, ...}, ...}
        print('Dump scores in ', 'cache/cosine_sem_we.pkl')
        pickle.dump(similarity_scores_we, open('cache/cosine_sem_we.pkl', 'wb'))
        # print(similarity_scores_we)

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

    def generate_features(self):
        print('================== Calculate BM25 scores  ===================')
        # self.calculate_bm25()

        print('================== Cosine Similarity scores  ===================')
        self.calculate_cosine_tf_idf()

        print('================== Cosine Similarity scores with query expansion  ===================')
        self.calculate_cosine_tf_idf_query_expansion()

        print('================== Cosine Similarity scores with semantic word embeddings  ===================')
        self.calculate_cosine_semantic_embeddings()

        print('================== Create DataFrame with Features  ===================')


print('================== Load Data ===================')
feature_generator = FeatureGenerator()

feature_generator.create_cache()

feature_generator.generate_features()

# queries = pickle.load(open('processed_data/lemma_processed_query.pkl', 'rb'))
# print(queries)

bm25 = pickle.load(open('cache/bm25_scores.pkl', 'rb'))


tfidfem = pickle.load(open('cache/cosine_sem_we.pkl', 'rb'))


tfidf = pickle.load(open('cache/cosine_tf_idf.pkl', 'rb'))

query_expansion = pickle.load(open('cache/cosine_query_expansion.pkl', 'rb'))

print(bm25['activity theory'])
print(tfidf['activity theory'])
print(tfidfem['activity theory'])
print(query_expansion['activity theory'])

