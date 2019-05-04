# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'

import numpy as np

from utils import Utils

class Performance:
    """
    This class contains some methods to calculate the performance.
    Also it contains some methods to process dicts.
    """

    ########################
    #   Some methods necessary to retrieve metrics
    ########################

    @staticmethod
    def get_true_positives(total_retrieved, actual_doc_ids, actual_relevance):
        """

        :param total_retrieved: documents retrieved with > 0 relevance score
        :param actual_doc_ids: array of doc ids (sorted corresponding to actual_relevance array)
        :param actual_relevance:  array of relevance scores (sorted corresponding to actual_doc_ids array)
        :return: number of true positives
        """
        true_positive = 0
        for doc, score in total_retrieved.items():
            # actual is dataframe with columns docid,
            try:
                if actual_relevance[actual_doc_ids.index(doc)] > 0:
                    true_positive += 1
            except ValueError:
                pass
        return true_positive

    @staticmethod
    def get_false_negatives(total_pred_negative, actual_doc_ids, actual_relevance):
        """

        :param total_pred_negative:
        :param actual_doc_ids:
        :param actual_relevance:
        :return:
        """
        false_negative = 0
        for doc, score in total_pred_negative.items():
            try:
                if actual_relevance[actual_doc_ids.index(doc)] > 0:
                    false_negative += 1
            except ValueError:
                pass
        return false_negative

    @staticmethod
    def filter_relevance_by_threshold(scores, threshold=0.0):
        """

        :param scores: dict of key, value pairs
        :param threshold: specify a threshold if necessary, threshold is 0.0 by default
        :return: filtered dict only containing scores > threshold
        """
        filtered = {}
        for k, v in scores.items():
            if v > threshold:
                filtered.update({k: v})
        return filtered

    @staticmethod
    def filter_pred_negative(scores):
        """

        :param scores: dict of key, value pairs
        :return: filtered dict only containing scores = 0
        """
        filtered = {}
        for k, v in scores.items():
            if v == 0:
                filtered.update({k: v})
        return filtered

    @staticmethod
    def filter_relevance_by_top_k(scores, top_k=100):
        """

        :param scores: dict of key, value pairs
        :param top_k: top_k has default value of 100
        :return: filtered dict of top_k best scores
        """
        filtered = {}
        index = 0
        for w in sorted(scores, key=scores.get, reverse=True):
            if index < top_k:
                filtered.update({w: scores[w]})
                index += 1
            else:
                break
        return filtered


class AveragePrecision:

    def __init__(self):
        self.precision_scores = {}
        self.recall_scores = {}

    def add_precision_score(self, q, precision):
        self.precision_scores.update({q: precision})

    def add_recall_score(self, q, recall):
        self.recall_scores.update({q: recall})

    def mean_avg_precision(self):
        map_score = 0.0
        len_prec = len(self.precision_scores.keys())
        if len_prec > 0:
            for k, v in self.precision_scores.items():
                map_score += v
            return map_score/len_prec
        else:
            return 0

    @staticmethod
    def avg_precision_score(predicted, actual, threshold=0, top_k=-1):
        """

        :param predicted: dict of predicted {docid: score, ...}
        :param actual: dataframe of actual to corresponding query
        :param threshold:
        :param top_k:
        :return: avg precision score corresponding to predicted and actual labels
        """
        # contains only documents with < threshold relevance
        total_retrieved = Performance.filter_relevance_by_threshold(predicted, threshold=threshold)

        # retrieves best top_k relevant documents
        if top_k > 0:
            total_retrieved = Performance.filter_relevance_by_top_k(predicted, top_k)

        # actual values of labeled set
        # arrays actual_doc_ids and actual_relevance are mapped by there indices
        # like this {actual_doc_ids[i]: actual_relevance[i], ...} for all i in range
        actual_doc_ids = list(actual['docid'])
        actual_relevance = list(actual['rel'])

        count = 0
        precision_arr = []

        for index, docs in enumerate(sorted(total_retrieved, key=total_retrieved.get, reverse=True)):
            try:
                if actual_relevance[actual_doc_ids.index(docs)] == 1:
                    count += 1
                    precision_arr.append(count/(index+1))
            except ValueError:
                pass

        ap_score = 0
        if len(precision_arr) > 0:
            for prec in precision_arr:
                ap_score += prec
            ap_score = ap_score/len(precision_arr)
        return ap_score

    # precision and recall values as input
    @staticmethod
    def f1_score(pre, rec):
        return 2*(pre * rec) / (pre + rec)

    def calculate_map(self, name, queries, true_labels, predicted_scores, top_k, threshold=0):
        """

        :param name: scores to be evaluated (bm25, tf-idf, ...)
        :param queries: all queries
        :param true_labels: true labels of query and corresponding doc
        :param predicted_scores: predicted relevances
        :param top_k: limit to top relevant document
        :param threshold: minimum score for doc to be considered relevant
        :return: map score
        """
        print('Calculate MAP for:', name)
        for q in queries:
            y_true = Utils.get_doc_and_relevance_by_query(q, true_labels)
            try:
                y_pred = predicted_scores[q]
                precision = AveragePrecision.avg_precision_score(y_pred, y_true, top_k=top_k, threshold=threshold)
                # recall = AveragePrecision.avg_recall_score(y_pred, y_true)
                self.add_precision_score(q, precision)
                # a.add_recall_score(q, recall)
            except KeyError:
                pass
        map_score = self.mean_avg_precision()
        print(name, 'MAP ', map_score)
        return map_score


class ReciprocalRank:

    def __init__(self):
        self.ranks = []

    @staticmethod
    def rank(predicted, actual, threshold=0, top_k=-1):
        """

        :param predicted: dict of predicted scores {docid: score}
        :param actual: df of true labels
        :param threshold: filter threshold score
        :param top_k: take best top_k relevant docs
        :return: returns the reciprocal rank of the predicted query document
        """
        total_retrieved = Performance.filter_relevance_by_threshold(predicted, threshold=threshold)

        # retrieves best top_k relevant documents
        if top_k > 0:
            total_retrieved = Performance.filter_relevance_by_top_k(predicted, top_k)

        actual_doc_ids = list(actual['docid'])
        actual_relevance = list(actual['rel'])

        rank = 0

        for index, docs in enumerate(sorted(total_retrieved, key=total_retrieved.get, reverse=True)):
            try:
                if actual_relevance[actual_doc_ids.index(docs)] == 1:
                    rank = 1/(index + 1)
                    break
            except ValueError:
                pass
        return rank

    def add_rank(self, rank):
        self.ranks.append(rank)

    def mean_reciprocal_rank(self):
        num_queries = len(self.ranks)
        return np.sum(self.ranks)/num_queries

    def calculate_mrr(self, name, queries, true_labels, predicted_scores, top_k, threshold=0):
        """

        :param name: scores to be evaluated (bm25, tf-idf, ...)
        :param queries: all queries
        :param true_labels: true labels of query and corresponding doc
        :param predicted_scores: predicted relevances
        :param top_k: limit to top relevant document
        :param threshold: minimum score for doc to be considered relevant
        :return: mrr score
        """
        print('Calculate MRR for:', name)
        for q in queries:
            y_true = Utils.get_doc_and_relevance_by_query(q, true_labels)
            try:
                y_pred = predicted_scores[q]
                reciprocal_rank = ReciprocalRank.rank(y_pred, y_true, top_k=top_k, threshold=threshold)
                self.add_rank(reciprocal_rank)
            except KeyError:
                pass
        mrr_score = self.mean_reciprocal_rank()
        print(name, 'MRR ', mrr_score)
        return mrr_score





