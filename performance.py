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
    def filter_relevance_by_top_k(scores, top_k=20):
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

    @staticmethod
    def filter_ranked_list(scores, top_k, threshold):
        """

        :param scores:
        :param top_k: use filter_relevance_by_top_k
        :param threshold: use filter_relevance_by_threshold
        :return: a dict filtered by top_k and threshold
        """
        total_retrieved = Performance.filter_relevance_by_threshold(scores, threshold=threshold)
        if top_k > 0:
            total_retrieved = Performance.filter_relevance_by_top_k(scores, top_k)

        return total_retrieved

    @staticmethod
    def filter_docs_in_actual(actual_doc_ids, predicted):
        """

        :param actual_doc_ids: list of docids that are in the set of actual labels
        :param predicted: dictionary with predicted relevance scores.
        :return: dictionary only containing values with docids that are also in the test set with actual labels.
        """
        total_retrieved = {}

        for index, docid in enumerate(actual_doc_ids):
            try:
                total_retrieved.update({docid: predicted[docid]})
            except KeyError:
                pass

        return total_retrieved


class AveragePrecision:

    def __init__(self):
        self.precision_scores = {}

    def add_precision_score(self, q, precision):
        self.precision_scores.update({q: precision})

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
    def avg_precision_score(predicted, actual, threshold=0, top_k=-1, only_actual=True):
        """

        :param predicted: dict of predicted {docid: score, ...}
        :param actual: dataframe of actual to corresponding query
        :param threshold:
        :param top_k:
        :param only_actual: true, then only use docs that are also in actual set,
        else apply the filter with top_k and threshold.
        :return: avg precision score corresponding to predicted and actual labels
        """
        # contains only documents with < threshold relevance
        # total_retrieved = Performance.filter_ranked_list(predicted, top_k, threshold)

        # actual values of labeled set
        # arrays actual_doc_ids and actual_relevance are mapped by there indices
        # like this {actual_doc_ids[i]: actual_relevance[i], ...} for all i in range
        actual_doc_ids = list(actual['docid'])
        actual_relevance = list(actual['rel'])

        if not only_actual:
            total_retrieved = Performance.filter_ranked_list(predicted, top_k, threshold)
        else:
            total_retrieved = Performance.filter_docs_in_actual(actual_doc_ids, predicted)

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

    def calculate_map(self, name, queries, true_labels, predicted_scores, top_k, threshold=0, only_actual=True):
        """

        :param name: scores to be evaluated (bm25, tf-idf, ...)
        :param queries: all queries
        :param true_labels: dataframe true labels of query and corresponding doc
        :param predicted_scores: predicted relevances
        :param top_k: limit to top relevant document
        :param threshold: minimum score for doc to be considered relevant
        :return: map score
        """
        for q in queries:
            y_true = Utils.get_doc_and_relevance_by_query(q, true_labels)
            try:
                y_pred = predicted_scores[q]
                precision = AveragePrecision.avg_precision_score(y_pred, y_true, top_k=top_k, threshold=threshold, only_actual=only_actual)
                # recall = AveragePrecision.avg_recall_score(y_pred, y_true)
                self.add_precision_score(q, precision)
                # a.add_recall_score(q, recall)
            except KeyError:
                pass
        map_score = self.mean_avg_precision()
        print(name, 'MAP ', map_score)
        return map_score


class ReciprocalRank:
    """
    Class to calculate the Reciprocal Rank.
    """

    def __init__(self):
        self.ranks = []

    @staticmethod
    def rank(predicted, actual, threshold=0, top_k=-1, only_actual=True):
        """

        :param predicted: dict of predicted scores {docid: score}
        :param actual: df of true labels
        :param threshold: filter threshold score
        :param top_k: take best top_k relevant docs
        :param only_actual: true, then only use docs that are also in actual set,
        else apply the filter with top_k and threshold.
        :return: returns the reciprocal rank of the predicted query document
        """

        actual_doc_ids = list(actual['docid'])
        actual_relevance = list(actual['rel'])

        if not only_actual:
            total_retrieved = Performance.filter_ranked_list(predicted, top_k, threshold)
        else:
            total_retrieved = Performance.filter_docs_in_actual(actual_doc_ids, predicted)

        rank = 0

        for index, docs in enumerate(sorted(total_retrieved, key=total_retrieved.get, reverse=True)):
            # search for first document that is true positive and use its retrieved index for its reciprocal rank
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

    def calculate_mrr(self, name, queries, true_labels, predicted_scores, top_k, threshold=0, only_actual=True):
        """

        :param name: scores to be evaluated (bm25, tf-idf, ...)
        :param queries: all queries
        :param true_labels: dataframe true labels of query and corresponding doc
        :param predicted_scores: predicted relevances
        :param top_k: limit to top relevant document
        :param threshold: minimum score for doc to be considered relevant
        :return: mrr score
        """
        for q in queries:
            y_true = Utils.get_doc_and_relevance_by_query(q, true_labels)
            try:
                y_pred = predicted_scores[q]
                reciprocal_rank = ReciprocalRank.rank(y_pred, y_true, top_k=top_k, threshold=threshold, only_actual=only_actual)
                self.add_rank(reciprocal_rank)
            except KeyError:
                pass
        mrr_score = self.mean_reciprocal_rank()
        print(name, 'MRR ', mrr_score)
        return mrr_score


class Precision:
    """
    Class to calculate R-Precision.
    """

    def __init__(self):
        self.scores = []

    @staticmethod
    def r_prec(predicted, actual, threshold=0, top_k=-1, only_actual=True):
        """

        :param predicted: predicted doc relevance
        :param actual: actual relevance
        :param threshold:
        :param top_k:
        :param only_actual: true, then only use docs that are also in actual set,
        else apply the filter with top_k and threshold.
        :return: R-Prec
        """
        actual_doc_ids = list(actual['docid'])
        actual_relevance = list(actual['rel'])
        if not only_actual:
            total_retrieved = Performance.filter_ranked_list(predicted, top_k, threshold)
        else:
            total_retrieved = Performance.filter_docs_in_actual(actual_doc_ids, predicted)

        r_prec = 0
        counter = 0  # count actual relevant
        for index, docs in enumerate(sorted(total_retrieved, key=total_retrieved.get, reverse=True)):
            try:
                if actual_relevance[actual_doc_ids.index(docs)] == 1:
                    counter += 1
            except ValueError:
                pass
        num_relevant = np.sum(actual_relevance)
        if num_relevant > 0:
            r_prec = counter/num_relevant

        return r_prec

    def mean_r_prec(self):
        num_r_prec = len(self.scores)
        return np.sum(self.scores)/num_r_prec

    def calculate_r_prec(self, name, queries, true_labels, predicted_scores, top_k, threshold=0, only_actual=True):
        """

        :param name: scores to be evaluated (bm25, tf-idf, ...)
        :param queries: all queries as list
        :param true_labels: dataframe true labels of query and corresponding doc
        :param predicted_scores: predicted relevances
        :param top_k: limit to top relevant document
        :param threshold: minimum score for doc to be considered relevant
        :return: r-prec score
        """
        for q in queries:
            y_true = Utils.get_doc_and_relevance_by_query(q, true_labels)
            try:
                y_pred = predicted_scores[q]
                r_prec = Precision.r_prec(y_pred, y_true, threshold=threshold, top_k=top_k, only_actual=only_actual)
                self.scores.append(r_prec)
            except KeyError:
                pass
        r_prec_score = self.mean_r_prec()
        print(name, 'R-Prec', r_prec_score)
        return r_prec_score

class StandardMatrics:
    """
    Class for compute standard performance scores
    """
    def __init__(self, y_pred, y_true, threshold=0):
        self.threshold = threshold
        self.y_pred = y_pred
        self.y_true = y_true
        self.TP = self.get_tp()
        self.FN = self.get_fn()
        self.FP = self.get_fp()
        self.TN = self.get_tn()

    def get_tp(self):
        tp = 0
        for k, v in self.y_true.items():
            for i in v[0]:
                try:
                    if i in self.y_pred[k]:
                        tp = tp + 1
                except KeyError:
                    pass
        return tp

    def get_fn(self):
        fn = 0
        for k, v in self.y_true.items():
            for i in v[0]:
                try:
                    if i not in self.y_pred[k]:
                        fn = fn + 1
                except KeyError:
                    pass
        return fn

    def get_fp(self):
        fp = 0
        for k, v in self.y_pred.items():
            for i in list(v.keys()):
                try:
                    if i not in self.y_true[k][0]:
                        fp = fp + 1
                except KeyError:
                    pass
        return fp

    def get_tn(self):
        # tn = 0
        # for k, v in self.y_pred.items():
        #     for i in list(v.keys()):
        #         try:
        #             if i not in self.y_true[k][0]:
        #                 tn = tn + 1
        #         except KeyError:
        #             pass
        tn = (4055*4055)-self.TP-self.FP-self.FN
        return tn

    def get_matrix_value(self):
        tp = self.TP
        fn = self.FN
        tn = self.TN
        fp = self.FP
        print(f"tp:{tp}",
              f"fn:{fn}",
              f"tn:{tn}",
              f"fp:{fp}")
        return tp, fn , tn , fp

    def confusion_matrix(self):
        cm = np.array([[self.TP, self.FP], [self.FN, self.TN]])
        return cm

    def calculate_acc(self, name):
        acc = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        print(name, 'accuracy:', acc)
        return acc

    def calculate_precision(self, name):
        precision = self.TP / (self.TP + self.FP)
        print(name, 'Precision:', precision)
        return precision

    def calculate_recall(self, name):
        recall = self.TP / (self.FN + self.TP)
        print(name, 'Recall:', recall)
        return recall

    def calculate_f1(self, name):
        p = self.TP / (self.TP + self.FP)
        r = self.TP / (self.FN + self.TP)
        f1 = (2*p*r) / (p + r)
        print(name, 'F1:', f1)
        return f1

class StandardAverageMetrics:
    """
    Class for compute average standard performance scores
    """

    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.precision_scores={}
        self.recall_sccores={}
        self.f1_scores={}
        self.acc_scores={}

    def add_each_scores(self):

        for k, v in self.y_true.items():
            tp = 0
            fp = 0
            fn = 0
            total = len(self.y_true.keys())-1 # total docs number except itself
            for i in v[0]:
                try:
                    if i in self.y_pred[k].keys():
                        tp = tp + 1
                    else:
                        fn = fn + 1
                except KeyError:
                    pass

            try:
                for i in self.y_pred[k].keys():
                    try:
                        if i not in v[0]:
                            fp = fp + 1
                    except KeyError:
                        pass

                tn = total - tp - fp - fn
                try:
                    p = tp / (tp + fp)
                    self.precision_scores.update({k: p})

                    r = tp / (tp + fn)
                    self.recall_sccores.update({k: r})

                    acc = (tp + tn) / (tp + tn + fp + fn)
                    self.acc_scores.update({k: acc})

                    f1 = (2 * p * r) / (p + r)
                    self.f1_scores.update({k: f1})
                except ZeroDivisionError:
                    pass
            except KeyError:
                pass

    def calculate_avg_scores(self, name):
        self.add_each_scores()
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_acc = 0.0
        len_doc = len(self.y_true.keys())

        for k, v in self.precision_scores.items():
            total_precision += v

        for k, v in self.recall_sccores.items():
            total_recall += v

        for k, v in self.f1_scores.items():
            total_f1 += v

        for k, v in self.acc_scores.items():
            total_acc += v

        print(f"{name} Acc:{total_acc/len_doc}\n"
              f"{name} P:{total_precision/len_doc}\n"
              f"{name} R:{total_recall/len_doc}\n"
              f"{name} F1:{total_f1/len_doc}\n"
              )

        return total_acc/len_doc, total_precision/len_doc, total_recall/len_doc, total_f1/len_doc
