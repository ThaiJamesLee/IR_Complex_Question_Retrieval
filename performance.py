# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'


# class to measure performance
class Performance:

    # actual: Dataframe with columns query, q0, doc_id, relevance
    # predicted: dict {query: {doc_id: relevance}}
    #
    @staticmethod
    def precision_score(predicted, actual):
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

