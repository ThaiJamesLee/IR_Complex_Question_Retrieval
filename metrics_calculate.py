import pickle
import pandas as pd

from utils import Utils
from performance import *
from cache_embedding_vectors import Caching
from utils import Utils
from threading import Thread
from queue import Queue


class Metrics:

    def __init__(self, top_k=10):
        print('Load data...')

        # processed queries
        self.queries = pickle.load(open('process_data/lemma_processed_query.pkl', 'rb'))

        # true labels
        self.true_labels = pickle.load(open('process_data/process_test.pkl', 'rb'))

        # bm25 scores
        self.bm25_scores = pickle.load(open('cache/bm25_scores.pkl', 'rb'))

        # cosine similarity tfidf
        self.tfidf_scores = pickle.load(open('cache/cosine_tf_idf.pkl', 'rb'))

        # cosine similarity tfidf with rocchio
        self.tfidf_rocchio_scores = pickle.load(open('cache/cosine_query_expansion.pkl', 'rb'))

        # cosine similarity tfidf with glove word embedding
        self.glove_scores = pickle.load(open('cache/cosine_sem_we.pkl', 'rb'))

        # cosine similarity tfidf with glove word embedding + rocchio
        self.glove_rocchio_scores = pickle.load(open('cache/cosine_sem_we_query_exp.pkl', 'rb'))

        self.top_k = top_k
        print('Set documents top_k to', top_k)

        """
        Run the metrics calculation with multiple threads.
        The order of the output is not guaranteed.
        """
        self.batches = {'BM25': self.bm25_scores, 'TF-IDF': self.tfidf_scores, 'TF-IDF + Roccio': self.tfidf_rocchio_scores, 'GloVe': self.glove_scores, 'GloVe + Rocchio': self.glove_rocchio_scores}

    def calculate_metrics(self, name, scores, queue, threshold, only_actual):
        """
        Calculates the metrics for scores defined in the batches variable.
        :param name:
        :param scores:
        :param queue:
        :param threshold:
        :param only_actual:
        :return:
        """
        map = AveragePrecision().calculate_map(name, self.queries, self.true_labels, scores, self.top_k, threshold=threshold, only_actual=only_actual)
        mrr = ReciprocalRank().calculate_mrr(name, self.queries, self.true_labels, scores, self.top_k, threshold=threshold, only_actual=only_actual)
        r_prec = Precision().calculate_r_prec(name, self.queries, self.true_labels, scores, self.top_k, threshold=threshold, only_actual=only_actual)
        if queue is not None:
            queue.put((map, mrr, r_prec))
        return map, mrr, r_prec


    def excecute_multithreaded(self, threshold=0.0, only_actual=False):
        """
        Create batches of tasks. Each batch calculates the R-Prec, MRR, and MAP score.
        :return:
        """
        score_queue = Queue()
        scores = []
        threads = []

        try:
            for k, v in self.batches.items():
                t = Thread(target=self.calculate_metrics, args=(k, v, score_queue, threshold, only_actual))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            while not score_queue.empty():
                scores.append(score_queue.get())
        except:
            print("Error: unable to start thread")
        return scores

    def execute_singethreaded(self, threshold=0.0, only_actual=False):
        """

        :return: Return tuples of scores.
        """
        scores = []
        for k, v in self.batches.items():
            scores.append(self.calculate_metrics(k, v, threshold, only_actual))

        return scores

    def calculate_map_bm25(self, name, threshold=0.0, only_actual=True):
        """
        Used for some testing.
        :param name:
        :param threshold:
        :param only_actual:
        :return:
        """
        map = AveragePrecision().calculate_map(name, self.queries, self.true_labels, self.bm25_scores, self.top_k, threshold=threshold, only_actual=only_actual)
        return map

class Standard:
    def __init__(self):
        print('load doc_doc data...')

        self.true_label = pickle.load(open('documents_retrieval/doc_rel.pkl', 'rb'))
        self.bm25 = pickle.load(open('documents_retrieval/doc_doc_bm25_scores.pkl', 'rb'))
        self.glove = pickle.load(open('documents_retrieval/doc_doc_glove_scores.pkl', 'rb'))
        self.glove_rocchio = pickle.load(open('documents_retrieval/doc_doc_glove_rocchio_scores_5.pkl', 'rb'))
        self.tfidf = pickle.load(open('documents_retrieval/doc_doc_tfidf_scores.pkl', 'rb'))
        self.tfidf_rocchio = pickle.load(open('documents_retrieval/doc_doc_tfidf_rocchio_scores_5.pkl', 'rb'))

        self.batches = {'BM25': self.bm25, 'TF-IDF': self.tfidf, 'TF-IDF + Roccio': self.tfidf_rocchio, 'GloVe': self.glove, 'GloVe + Rocchio': self.glove_rocchio}


    def calculate_standard_metrics(self, name, scores, queue, threshold):
        """
        Calculates the standard metrics for document classification.
        :param predicted: array of predicted label
        :param actual: array of true label
        :return: acc, P, R, F1
        """
        m = StandardMatrics(y_pred=scores, y_true=self.true_label, threshold=threshold)
        tp, fn, tn, fp = m.get_matrix_value()
        acc = m.calculate_acc(name)
        p = m.calculate_precision(name)
        r = m.calculate_recall(name)
        f1 = m.calculate_f1(name)
        if queue is not None:
            queue.put((acc, p, r, f1))
        return acc, p, r, f1

    def excecute_stand_multithreaded(self, threshold=0.0):
        """
        Create batches of tasks. Each batch calculates the R-Prec, MRR, and MAP score.
        :return:
        """
        score_queue = Queue()
        scores = []
        threads = []

        try:
            for k, v in self.batches.items():
                t = Thread(target=self.calculate_standard_metrics, args=(k, v, score_queue, threshold))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            while not score_queue.empty():
                scores.append(score_queue.get())
        except:
            print("Error: unable to start thread")
        return scores

    def calculate_stand_bm25(self, name, threshold=0.0):
        """
        Used for some testing.
        :param name:
        :param threshold:
        :param only_actual:
        :return:
        """
        print('threshold:', threshold)
        m = StandardMatrics(y_pred=self.bm25, y_true=self.true_label, threshold=threshold)
        tp, fn, tn, fp = m.get_matrix_value()
        acc = m.calculate_acc(name)
        p = m.calculate_precision(name)
        r = m.calculate_recall(name)
        f1 = m.calculate_f1(name)
        return acc, p, r, f1

    def calculate_stand_tfidf(self, name, threshold=0.0):
        """
        Used for some testing.
        :param name:
        :param threshold:
        :param only_actual:
        :return:
        """
        print('threshold:', threshold)
        m = StandardMatrics(y_pred=self.tfidf, y_true=self.true_label, threshold=threshold)
        tp, fn, tn, fp = m.get_matrix_value()
        acc = m.calculate_acc(name)
        p = m.calculate_precision(name)
        r = m.calculate_recall(name)
        f1 = m.calculate_f1(name)
        return acc, p, r, f1

    def calculate_stand_glove(self, name, threshold=0.0):
        """
        Used for some testing.
        :param name:
        :param threshold:
        :param only_actual:
        :return:
        """
        print('threshold:', threshold)
        m = StandardMatrics(y_pred=self.glove, y_true=self.true_label, threshold=threshold)
        tp, fn, tn, fp = m.get_matrix_value()
        acc = m.calculate_acc(name)
        p = m.calculate_precision(name)
        r = m.calculate_recall(name)
        f1 = m.calculate_f1(name)
        return acc, p, r, f1

# m = Metrics(top_k=20)
# print(m.calculate_map_bm25('BM25'))
# print(Metrics(top_k=30).excecute_multithreaded(only_actual=True))
# print(execute_singethreaded())


m = Standard()
# run all scores with same threshold..
# print(m.excecute_stand_multithreaded())

# tune bm25 threshold ..
print(m.calculate_stand_bm25('BM25',threshold=0))
print(m.calculate_stand_bm25('BM25',threshold=10))

# tune tf-idf threshold..
print(m.calculate_stand_tfidf('TF-IDF',threshold=0.0))
print(m.calculate_stand_tfidf('TF-IDF',threshold=0.5))

# tune glove threshold..
print(m.calculate_stand_glove('GloVe',threshold=0.0))
print(m.calculate_stand_glove('GloVe',threshold=0.5))