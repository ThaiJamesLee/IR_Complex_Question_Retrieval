# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'
import numpy as np


class Performance:
    """
    This class contains some methods to calculate the performance.
    Also it contains some methods to process dicts.
    """

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
            raise Exception('No precision scores available!')

    @staticmethod
    def avg_precision_score(predicted, actual):
        """

        :param predicted: dict of predicted {docid: score, ...}
        :param actual: dataframe of actual to corresponding query
        :return: avg precision score corresponding to predicted and actual labels
        """
        # contains only documents with < 0 relevance
        total_retrieved = Performance.filter_relevance_by_threshold(predicted)

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

    @staticmethod
    def avg_recall_score(predicted, actual):
        """

        :param predicted: predicted: dict of predicted {docid: score, ...}
        :param actual: actual: dataframe of actual to corresponding query
        :return: avg recall score corresponding to predicted and actual labels
        """
        # contains only documents with < 0 relevance
        total_retrieved = Performance.filter_relevance_by_threshold(predicted)

        actual_doc_ids = list(actual['docid'])
        actual_relevance = list(actual['rel'])
        # number of relevant documents
        # total_pred_negative = Performance.filter_pred_negative(predicted)
        # false_negative = Performance.get_false_negatives(total_pred_negative, actual_doc_ids, actual_relevance)
        no_relevant_doc = 0
        for rel in actual_relevance:
            no_relevant_doc += rel

        count = 0
        recall_arr = []
        for index, docs in enumerate(sorted(total_retrieved, key=total_retrieved.get, reverse=True)):
            try:
                if actual_relevance[actual_doc_ids.index(docs)] == 1:
                    count += 1
                    recall_arr.append(count / no_relevant_doc)
            except ValueError:
                pass

        ar_score = 0
        if len(recall_arr) > 0:
            for rec in recall_arr:
                ar_score += rec
            ar_score = ar_score/len(recall_arr)
        return ar_score

    # precision and recall values as input
    @staticmethod
    def f1_score(pre, rec):
        return 2*(pre * rec) / (pre + rec)

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
