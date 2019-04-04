from rocchio import RocchioOptimizeQuery

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



