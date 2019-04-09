from rocchio import RocchioOptimizeQuery

# from performance import Performance
# from bm25 import BM25

print('================== Implementation of Rocchio ==================')
print('')

matrix = {'1': {'apple': 0.1, 'peach': 0.2}, '2': {'apple': 0.4, 'orange': 0.5},
          '3': {'apple': 0.5, 'peach': 0.3}, '4': {'peach': 0.4, 'banana': 0.2}}

relevant_docs = [1, 2, 3]
non_relevant_docs = [4]

query_vector = {'apple': 0.5, 'peach': 0.5}

rocchio = RocchioOptimizeQuery(initial_query_vector=query_vector, tf_idf_matrix=matrix)
new_query_vector = rocchio.execute_rocchio(relevant_docs, non_relevant_docs)
print('initial query : ', query_vector)
print('expanded query : ', new_query_vector)

# print('================== Implementation of Rocchio with BM25 ==================')
# print('')
# # given rel_scores from BM25
# # given tf-idf matrix
#
# bm25 = BM25(doc_structure, k=1.2)
# p = Performance()



