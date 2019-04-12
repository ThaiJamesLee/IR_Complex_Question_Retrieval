# 1. Load data or run data preprocess, if use existing -> load data, if new dataset, then run preprocess
# --> dataframe for test and train, run doc_structure function for input bm25
# 2. calculate bm25 --> candidates
# --> dict {query: {docid: score, ...}, ...}
# --> dump as object, use pickle
# 3. format to tf-idf
# --> dict {docid: {term: value, ...}, ...}
# 4. query expansion with tf-idf
# 5. merge results of bm25, tf-idf (cosine), tf-idf+query expansion (cosine) --> new dataframe
# 6. put into L2R and Duet Models
# 7. evaluate (bm25, tf_idf(+query expansion), L2R, Duet) using MAP, R-Prec, MRR
# 8. create new query and visualize result

