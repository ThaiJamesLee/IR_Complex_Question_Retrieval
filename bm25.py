# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd

__author__ = 'Duc Tai Ly'


class BM25:

    def __init__(self, documents, b=None, k=None):
        """

        see: https://www.researchgate.net/publication/220613776_The_Probabilistic_Relevance_Framework_BM25_and_Beyond
        :param documents: is a dict of doc_ids containing terms with corresponding weights
        :param b:  default: b = 0.75, b is best in [0.5, 0.8]
        :param k:  default: k = 1.2, k is best in [1.2, 2.0]
        """
        self.documents = documents
        self.cache = dict()
        self.num_docs = len(self.documents.keys())
        self.avg_doc_len = self.get_average_doc_length()
        if b is None:
            self.b = 0.75
        else:
            self.b = b
        if k is None:
            self.k = 1.2
        else:
            self.k = k

    def get_doc_length(self, doc_id):
        """

        :param doc_id: Document Id.
        :return: document exact length of specified doc_id
        """
        return np.sum(list(self.documents[doc_id].values()))

    def get_average_doc_length(self):
        """
        :return: documents average length
        """
        num_docs = self.num_docs
        num_items = 0

        for k, v in self.documents.items():
            num_items = num_items + len(v)
        return num_items / num_docs

    def get_term_frequency_in_doc(self, term, doc_id):
        """
        if the term is not in the document, then we will get an KeyError eventually --> runtime O(1)
        :param term: Query term.
        :param doc_id: Document id.
        :return: get frequency of term in document with doc_id
        """
        try:
            return self.documents[doc_id][term]
        except KeyError:
            return 0

    def idf_weight(self, term):
        """
         idf = log10(#docs/df)
        :param term:  get weight of term as idf weight
        :return: the idf weight for given term.
        """
        df = 0
        num_docs = len(self.documents.keys())
        for k, v in self.documents.items():
            if self.get_term_frequency_in_doc(term, k) > 0:
                df = df + 1
        if df != 0:
            return math.log(num_docs/df, 10)
        else:
            return 0

    def idf_weight_2(self, term):
        """
        see:  https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables
        idf = ln((1+(#docs-df+0.5))/())
        :param term: get weight based on article
        :return: the idf weight for given term.
        """
        df = 0
        if term not in self.cache.keys():

            for k, v in self.documents.items():
                if self.get_term_frequency_in_doc(term, k) > 0:
                    df = df + 1
            self.cache.update({term: df})
        else:
            df = self.cache[term]

        idf = math.log(1 + (self.num_docs - df + 0.5)/(df + 0.5))
        return idf

    def relevance(self, doc_id, query):
        """
         calculate relevance score of query and corresponding document
        :param doc_id: Document id
        :param query: terms in query should only be separated by blank spaces
        :return: the bm25 relevance score of the doc, query pair
        """
        score = 0
        terms = query

        doc_length = self.get_doc_length(doc_id)
        avg_doc_length = self.avg_doc_len

        doc_len_normalized = doc_length/avg_doc_length

        for term in terms:
            term_freq = self.get_term_frequency_in_doc(term, doc_id)
            # we ignore terms without term frequency --> improve performance significantly
            if term_freq > 0:
                counter = term_freq * (self.k + 1)
                denominator = term_freq + self.k * (1 - self.b + self.b * doc_len_normalized)
                idf = self.idf_weight_2(term)
                # relevance score of term in current document
                term_score = idf * counter/denominator
                score = score + term_score
        return score

    def compute_relevance_on_corpus(self, query):
        """

        computes relevance score for each document, with given query.
        If the score is 0, then skip.
        :param query: input as string separated by whitespaces
        :return: dict of {docid: relevance_score, ...}
        """
        if type(query) == 'str':
            query = query.split()
        elif type(query) != str and type(query) != list:
            raise Exception('Invalid input. The query must be of type str or list!')

        scores = {}
        for doc_id in self.documents.keys():
            score = self.relevance(doc_id, query)
            scores.update({doc_id: score})
        return scores
