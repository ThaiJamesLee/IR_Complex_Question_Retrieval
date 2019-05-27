# -*- coding: utf-8 -*-
__author__ = 'Ching Han Chen'
# this is used to test implementations
# and to show how to use the implemented code

from query_expansion.rocchio import RocchioOptimizeQuery
import pickle
from performance import Performance


# print('==================Testing Implementation of Rocchio ==================')
# print('')
#
# matrix = {'1': {'apple': 0.1, 'peach': 0.2}, '2': {'apple': 0.4, 'orange': 0.5},
#           '3': {'apple': 0.5, 'peach': 0.3}, '4': {'peach': 0.4, 'banana': 0.2}}
#
# relevant_docs = [1, 2, 3]
# non_relevant_docs = [4]
#
# query_vector = {'apple': 0.5, 'peach': 0.5}
#
# rocchio = RocchioOptimizeQuery(query_vector=query_vector, tf_idf_matrix=matrix)
# new_query_vector = rocchio.execute_rocchio(relevant_docs, non_relevant_docs)
# print('initial query : ', query_vector)
# print('expanded query : ', new_query_vector)

print('================== Implementation of Rocchio with BM25 ==================')

bm25 = pickle.load(open("processed_data/sample_input_bm25.pkl", 'rb'))
tf_idf = pickle.load(open("processed_data/sample_input_tfidf.pkl", 'rb'))

raw_query = 'asm'
query_vector = {'asm': 0.5, 'engin': 0.2}

p = Performance()
rocchio = RocchioOptimizeQuery(query_vector=query_vector, tf_idf_matrix=tf_idf)

relevant_docs = p.filter_relevance_by_top_k(bm25['asm'],10)
non_relevant_docs = p.filter_pred_negative(bm25['asm'])

new_query_vector = rocchio.execute_rocchio(relevant_docs, non_relevant_docs, number_new_terms=3)
expanded_new_terms = rocchio.get_topk_new_terms(5)
print('initial query : ', query_vector)
print('expanded query : ', new_query_vector)
print('new terms : ', expanded_new_terms)

