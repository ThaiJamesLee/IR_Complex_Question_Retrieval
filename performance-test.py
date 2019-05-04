import pickle

from utils import Utils
from performance import *
from cache_embedding_vectors import Caching
from utils import Utils

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

print('Calculate MAP scores...')
AveragePrecision().calculate_map('BM25', queries, true_labels, bm25_scores, top_k)
AveragePrecision().calculate_map('TF-IDF', queries, true_labels, tfidf_scores, top_k)
AveragePrecision().calculate_map('TF-IDF + Roccio', queries, true_labels, tfidf_rocchio_scores, top_k)
AveragePrecision().calculate_map('GloVe', queries, true_labels, glove_scores, top_k)
AveragePrecision().calculate_map('GloVe + Rocchio', queries, true_labels, glove_rocchio_scores, top_k)

print('Calculate MRR scores...')
ReciprocalRank().calculate_mrr('BM25', queries, true_labels, bm25_scores, top_k)
ReciprocalRank().calculate_mrr('TF-IDF', queries, true_labels, tfidf_scores, top_k)
ReciprocalRank().calculate_mrr('TF-IDF + Roccio', queries, true_labels, tfidf_rocchio_scores, top_k)
ReciprocalRank().calculate_mrr('GloVe', queries, true_labels, glove_scores, top_k)
ReciprocalRank().calculate_mrr('GloVe + Rocchio', queries, true_labels, glove_rocchio_scores, top_k)

# print(true_labels)

# c = Caching()
# c.create_document_embeddings()
# c.create_query_embeddings()
# c.create_query_embeddings_query_expansion()
# print(glove_rocchio_scores.keys())



