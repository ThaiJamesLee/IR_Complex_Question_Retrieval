import pickle

from utils import Utils
from performance import *
from cache_embedding_vectors import Caching
from utils import Utils
from threading import Thread

print('Load data...')

# processed queries
queries = pickle.load(open('process_data/lemma_processed_query.pkl', 'rb'))

# true labels
true_labels = pickle.load(open('process_data/process_test.pkl', 'rb'))

# bm25 scores
bm25_scores = pickle.load(open('cache/bm25_scores.pkl', 'rb'))

# cosine similarity tfidf
tfidf_scores = pickle.load(open('cache/cosine_tf_idf.pkl', 'rb'))

# cosine similarity tfidf with rocchio
tfidf_rocchio_scores = pickle.load(open('cache/cosine_query_expansion.pkl', 'rb'))

# cosine similarity tfidf with glove word embedding
glove_scores = pickle.load(open('cache/cosine_sem_we.pkl', 'rb'))

# cosine similarity tfidf with glove word embedding + rocchio
glove_rocchio_scores = pickle.load(open('cache/cosine_sem_we_query_exp.pkl', 'rb'))

top_k = 20
print('Set documents top_k to', top_k)

"""
Run the metrics calculation with multiple threads.
The order of the output is not guaranteed.
"""
batches = {'BM25': bm25_scores, 'TF-IDF': tfidf_scores, 'TF-IDF + Roccio': tfidf_rocchio_scores, 'GloVe': glove_scores, 'GloVe + Rocchio': glove_rocchio_scores}


def calculate_metrics(name, scores, threshold=0.0):
    map = AveragePrecision().calculate_map(name, queries, true_labels, scores, top_k, threshold=threshold)
    mrr = ReciprocalRank().calculate_mrr(name, queries, true_labels, scores, top_k, threshold=threshold)
    r_prec = Precision().calculate_r_prec(name, queries, true_labels, scores, top_k, threshold=threshold)
    return map, mrr, r_prec


if __name__ == '__main__':
    try:
        for k, v in batches.items():
            Thread(target=calculate_metrics, args=(k, v)).start()
    except:
        print("Error: unable to start thread")

"""
Comment out the lines above for multithreading, if it is not desired.
Comment in the code below for single threaded execution.
"""

"""
for k, v in batches.items():
    calculate_metrics(k, v)
"""
# print(true_labels)

# c = Caching()
# c.create_document_embeddings()
# c.create_query_embeddings()
# c.create_query_embeddings_query_expansion()
# print(glove_rocchio_scores.keys())



