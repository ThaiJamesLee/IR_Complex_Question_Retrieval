import pickle

from utils import Utils
from performance import AveragePrecision
from cache_query_and_docs_as_embedding_vectors import Caching
from utils import Utils

print('Load data...')

# processed queries
queries = pickle.load(open('processed_data/lemma_processed_query.pkl', 'rb'))

# true labels
true_labels = pickle.load(open('processed_data/process_test.pkl', 'rb'))

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


def map_bm25():
    print('MAP BM25')
    avg_bm25 = AveragePrecision()
    for q in queries:
        y_true = Utils.get_doc_and_relevance_by_query(q, true_labels)
        try:
            y_pred = bm25_scores[q]
            precision = AveragePrecision.avg_precision_score(y_pred, y_true, top_k=top_k)
            recall = AveragePrecision.avg_recall_score(y_pred, y_true)
            avg_bm25.add_precision_score(q, precision)
            avg_bm25.add_recall_score(q, recall)
        except KeyError:
            pass
    print('BM25 MAP ', avg_bm25.mean_avg_precision())


def map_tfidf():
    print('TF-IDF')
    avg_tfidf = AveragePrecision()
    for q in queries:
        y_true = Utils.get_doc_and_relevance_by_query(q, true_labels)
        try:
            y_pred = tfidf_scores[q]
            precision = AveragePrecision.avg_precision_score(y_pred, y_true, top_k=top_k)
            recall = AveragePrecision.avg_recall_score(y_pred, y_true)
            avg_tfidf.add_precision_score(q, precision)
            avg_tfidf.add_recall_score(q, recall)
        except KeyError:
            pass
    print('MAP TF-IDF ', avg_tfidf.mean_avg_precision())


def map_tfidf_rocchio():
    print('TF-IDF + Rocchio')
    avg_tfidf = AveragePrecision()
    for q in queries:
        y_true = Utils.get_doc_and_relevance_by_query(q, true_labels)
        try:
            y_pred = tfidf_rocchio_scores[q]
            precision = AveragePrecision.avg_precision_score(y_pred, y_true, top_k=top_k)
            recall = AveragePrecision.avg_recall_score(y_pred, y_true)
            avg_tfidf.add_precision_score(q, precision)
            avg_tfidf.add_recall_score(q, recall)
        except KeyError:
            pass
    print('TF-IDF+Rocchio MAP ', avg_tfidf.mean_avg_precision())


def map_glove():
    print('GloVe')
    avg_tfidf = AveragePrecision()
    for q in queries:
        y_true = Utils.get_doc_and_relevance_by_query(q, true_labels)
        try:
            y_pred = glove_scores[q]
            precision = AveragePrecision.avg_precision_score(y_pred, y_true, threshold=0.4, top_k=top_k)
            recall = AveragePrecision.avg_recall_score(y_pred, y_true)
            avg_tfidf.add_precision_score(q, precision)
            avg_tfidf.add_recall_score(q, recall)
            # print(q, precision)
        except KeyError:
            pass
    print('GloVe MAP ', avg_tfidf.mean_avg_precision())


def map_glove_rocchio():
    print('GloVe + Rocchio')
    avg_tfidf = AveragePrecision()
    for q in queries:
        y_true = Utils.get_doc_and_relevance_by_query(q, true_labels)
        try:
            y_pred = glove_rocchio_scores[q]
            precision = AveragePrecision.avg_precision_score(y_pred, y_true, threshold=0.4, top_k=top_k)
            recall = AveragePrecision.avg_recall_score(y_pred, y_true)
            avg_tfidf.add_precision_score(q, precision)
            avg_tfidf.add_recall_score(q, recall)
            # print(q, precision)
        except KeyError:
            pass
    print('GloVe+Rocchio MAP ', avg_tfidf.mean_avg_precision())


map_bm25() #.38
map_tfidf() #.36
map_tfidf_rocchio()  # 0.3421341465887873
map_glove()  # 0.00564886884668323 (t=0.3), 0.006662115724958531 (t=0.1) -- 0.24886603930313425
map_glove_rocchio()  # 0.006962405308865148 (t=0.1) -- 0.25965134852520605 (t=0.1)
# print(true_labels)

# c = Caching()
# c.create_document_embeddings()
# c.create_query_embeddings()
# c.create_query_embeddings_query_expansion()
# print(glove_rocchio_scores.keys())



