# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'


class Performance:

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
                filtered.update({k, v})
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

    # actual: Dataframe with columns query, q0, doc_id, relevance
    # predicted: dict {query: {doc_id: relevance}}
    #
    @staticmethod
    def precision_score(predicted, actual):
        """

        :param predicted: dict of predicted {docid: score, ...}
        :param actual: dataframe of actual to corresponding query
        :return:
        """
        total_retrieved = Performance.filter_relevance_by_threshold(predicted)
        no_total_retrieved = len(total_retrieved.keys())

        actual.drop(columns='q0')
        pass

    # actual: Dataframe with columns query, q0, doc_id, relevance
    # predicted: dict {query: {doc_id: relevance}}
    #
    @staticmethod
    def recall_score(predicted, actual):
        pass

    # precision and recall values as input
    @staticmethod
    def f1_score(pre, rec):
        return 2*(pre * rec) / (pre + rec)

