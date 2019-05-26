import pickle
import pandas as pd

from utils import Utils
from performance import *
from cache_embedding_vectors import Caching
from utils import Utils
from threading import Thread
from queue import Queue


class Metrics:

    def __init__(self, top_k=10, rocchio_terms=5):
        """

        :param top_k: sets top_k documents for rocchio and also for metrics calculation to consider.
        :param rocchio_terms:
        """
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
        self.tfidf_rocchio_scores = pickle.load(open(f'cache/cosine_query_expansion_{rocchio_terms}.pkl', 'rb'))

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
        # self.batches = {'TF-IDF + Roccio': self.tfidf_rocchio_scores, 'GloVe + Rocchio': self.glove_rocchio_scores}

    def calculate_metrics(self, name, scores, queue, threshold, only_actual):
        """
        Calculates the metrics for scores defined in the batches variable.
        :param name:
        :param scores:
        :param queue:
        :param threshold: sets threshold, only set this, when all metrics are normalized
        :param only_actual: True, considers only docs that have true labels and ignore top_k, otherwise use top_k only
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
    def __init__(self, top_k=5, only_actual=True):
        print('load doc_doc data...')

        # doc_doc relevant true label
        self.true_label = pickle.load(open('documents_retrieval/doc_rel.pkl', 'rb'))

        # bm25 scores
        self.bm25 = pickle.load(open('documents_retrieval/doc_doc_bm25_scores.pkl', 'rb'))

        # glove scores
        self.glove = pickle.load(open('documents_retrieval/doc_doc_glove_scores.pkl', 'rb'))
        # glove + query expansion scores
        self.glove_rocchio = pickle.load(open('documents_retrieval/doc_doc_glove_rocchio_scores_5.pkl', 'rb'))

        # TF-IDF scores
        self.tfidf = pickle.load(open('documents_retrieval/doc_doc_tfidf_scores.pkl', 'rb'))
        # TF-IDF + query expansion scores
        self.tfidf_rocchio = pickle.load(open('documents_retrieval/doc_doc_tfidf_rocchio_scores_5.pkl', 'rb'))

        # when only_actual is not True, we retrieve top_k relevant docs on each predictor.
        self.top_k = top_k

        # e.g. when doc A contains 3 relevant docs, we retrieve top 3 predicted docs for evaluation.
        self.only_actual = only_actual
        if only_actual is True:
            print('only_actual: True')
        else:
            print('Set documents top_k to', top_k)

        self.batches = {'BM25': self.bm25, 'TF-IDF': self.tfidf, 'TF-IDF + Roccio': self.tfidf_rocchio, 'GloVe': self.glove, 'GloVe + Rocchio': self.glove_rocchio}

    def filter_predict_by_top_k(self, scores, top_k =10):
        """
        :param scores: predicted docs, dict of key, value{docid:socre, ..} pairs
        :param top_k: top_k has default value of 10
        :return: filtered dict of top_k best scores
        """
        filtered_pred = {}
        for k, v in scores.items():
            filtered = {}
            index = 0
            for w in sorted(v, key=v.get, reverse=True):
                if index < top_k:
                    filtered.update({w: v[w]})
                    index += 1
                else:
                    break
            filtered_pred.update({k: filtered})
        return filtered_pred

    def filter_predict_by_actual(self, scores):
        """
        :param scores: predicted scores, dict of key, value{docid:socre, ..} pairs
        :return: retrieve top n scores of predicted docids, where n equal to number of true_label
                 e.g. if true label d1 contains 2 rel docs, then retrieve 2 highest docs in predicts.
        """

        filtered_pred = {}
        for k, v in scores.items():
            filtered = {}
            index = 0
            for w in sorted(v, key=v.get, reverse=True):
                try:
                    if index < len(self.true_label[k][0]):
                        filtered.update({w: v[w]})
                        index += 1
                    else:
                        break
                except KeyError:
                    pass
            filtered_pred.update({k: filtered})
        return filtered_pred

    def calculate_standard_metrics(self, name, scores, queue, threshold=0):
        """
        Aggregated method : aggregate all, compute one P/R/F1

        :param name: name of scores, e.g. 'BM25', 'TF-IDF'
        :param scores: dicts of predicted scores, {docid:{{docid:value, ...}}...}
        :param queue:
        :return: acc, P, R, F1
        """
        if self.only_actual is True:
            y_pred = self.filter_predict_by_actual(scores=scores)
        else :
            y_pred = self.filter_predict_by_top_k(scores=scores, top_k=self.top_k)

        m = StandardMatrics(y_pred=y_pred, y_true=self.true_label, threshold=threshold)
        tp, fn, tn, fp = m.get_matrix_value()
        acc = m.calculate_acc(name)
        p = m.calculate_precision(name)
        r = m.calculate_recall(name)
        f1 = m.calculate_f1(name)
        if queue is not None:
            queue.put((acc, p, r, f1))
        return acc, p, r, f1

    def calculate_standard_avg_metrics(self, name, scores, queue):
        """
        Average method : compute P/R/F1 for each query, then average.

        :param name: name of scores, e.g. 'BM25', 'TF-IDF'
        :param scores: dicts of predicted scores, {docid:{{docid:value, ...}}...}
        :param queue:
        :return: acc, P, R, F1
        """
        if self.only_actual is True:
            y_pred = self.filter_predict_by_actual(scores=scores)
        else :
            y_pred = self.filter_predict_by_top_k(scores=scores, top_k=self.top_k)

        m = StandardAverageMetrics(y_pred=y_pred, y_true=self.true_label)
        acc,p,r,f1 = m.calculate_avg_scores(name)
        if queue is not None:
            queue.put((acc, p, r, f1))
        return acc, p, r, f1

    def excecute_stand_multithreaded(self, threshold=0.0):
        """
        For Aggregated method :

        Create batches of tasks. Each batch calculates the Precision, Recall, and F1 score.
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

    def excecute_avg_stand_multithreaded(self):
        """
        For Average method :
        Create batches of tasks. Each batch calculates the Precision, Recall, and F1 score.
        :return:
        """
        score_queue = Queue()
        scores = []
        threads = []

        try:
            for k, v in self.batches.items():
                t = Thread(target=self.calculate_standard_avg_metrics, args=(k, v, score_queue))
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
        Used for bm25 testing.
        :param name:
        :param threshold:
        :param only_actual:
        :return:
        """
        if self.only_actual is False:
            y_pred = self.filter_predict_by_top_k(scores=self.bm25, top_k=self.top_k)
        else :
            y_pred = self.filter_predict_by_actual(scores=self.bm25)

        print('threshold:', threshold)
        m = StandardMatrics(y_pred=y_pred, y_true=self.true_label, threshold=threshold)
        tp, fn, tn, fp = m.get_matrix_value()
        acc = m.calculate_acc(name)
        p = m.calculate_precision(name)
        r = m.calculate_recall(name)
        f1 = m.calculate_f1(name)
        return acc, p, r, f1

    def calculate_stand_tfidf(self, name, threshold=0.0):
        """
        Used for tf-idf testing.
        :param name:
        :param threshold:
        :param only_actual:
        :return:
        """
        if self.only_actual is False:
            y_pred = self.filter_predict_by_top_k(scores=self.tfidf, top_k=self.top_k)
        else :
            y_pred = self.filter_predict_by_actual(scores=self.tfidf)

        print('threshold:', threshold)
        m = StandardMatrics(y_pred=y_pred, y_true=self.true_label, threshold=threshold)
        tp, fn, tn, fp = m.get_matrix_value()
        acc = m.calculate_acc(name)
        p = m.calculate_precision(name)
        r = m.calculate_recall(name)
        f1 = m.calculate_f1(name)
        return acc, p, r, f1

    def calculate_stand_glove(self, name, threshold=0.0):
        """
        Used for glove testing.
        :param name:
        :param threshold:
        :param only_actual:
        :return:
        """
        if self.only_actual is False:
            y_pred = self.filter_predict_by_top_k(scores=self.glove, top_k=self.top_k)
        else :
            y_pred = self.filter_predict_by_actual(scores=self.glove)

        print('threshold:', threshold)
        m = StandardMatrics(y_pred=y_pred, y_true=self.true_label, threshold=threshold)
        tp, fn, tn, fp = m.get_matrix_value()
        acc = m.calculate_acc(name)
        p = m.calculate_precision(name)
        r = m.calculate_recall(name)
        f1 = m.calculate_f1(name)
        return acc, p, r, f1

# m = Metrics(top_k=20)
# print(m.excecute_multithreaded(only_actual=True))
# print(m.calculate_map_bm25('BM25'))
# print(Metrics(top_k=30).excecute_multithreaded(only_actual=True))
# print(execute_singethreaded())



"""
To compute p/r/f1 with aggregated method
"""
# m = Standard(only_actual=True)
# print(m.excecute_stand_multithreaded(threshold=0))


"""
To compute p/r/f1 with average method
"""
# m = Standard(only_actual=True)
# print(m.excecute_avg_stand_multithreaded())

# tune bm25 threshold ..
# print(m.calculate_stand_bm25('BM25',threshold=5, only_actual=False))

# # tune tf-idf threshold..
# print(m.calculate_stand_tfidf('TF-IDF',threshold=0.0))
# print(m.calculate_stand_tfidf('TF-IDF',threshold=0.5))



#
# # tune glove threshold..
# print(m.calculate_stand_glove('GloVe',threshold=0.1))
# print(m.calculate_stand_glove('GloVe',threshold=0.5))