# -*- coding: utf-8 -*-
__author__ = 'Ching Han Chen'
"""
    Class to execute Rocchio algorithm:
    for each unique term across all documents (from tf_idf_matrix) are passed into Rocchio

    Input :
    * query : initial query list of query strings
    * tf_idf_matrix : contains tf_idf values for each doc/term entry
    * alpha : weights of initial query
    * beta : weights of relevant docs
    * gamma : weights of non relevant docs

    Output :
    * new query vector close towards the relevant and away from non-relevant documents

"""
class RocchioOptimizeQuery:

    def __init__(self, query_vector, tf_idf_matrix, alpha=1, beta=0.75, gamma=0.15):
        self.query_vector = query_vector
        self.new_query_vector = {}
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.matrix = tf_idf_matrix
        self.weights = self.get_initial_weight(tf_idf_matrix)
        self.new_terms = {}

    # e.g. list of documents that bm25 scores more than 0.1 are considered relevant
    def get_initial_weight(self, matrix):
        weights = {}
        for doc_id in matrix.keys():
            doc = matrix[doc_id]
            for term in list(doc.keys()):
                weights[term] = 0.0
        return weights

    def get_doc_weights(self, docs):
        doc_weights = {}
        for doc_id in docs:
            try:
                doc = self.matrix[doc_id]
                for term in list(doc.keys()):
                    if term in doc_weights:
                        doc_weights[term] = doc_weights[term] + doc[term]
                    else:
                        doc_weights[term] = doc[term]
            except KeyError:
                pass
        return doc_weights

    # number_new_terms : amount of terms wanted to be added into initial query
    # return : a new query vector which is origin query vector with additional added new terms vector.
    def execute_rocchio(self, relevant_docs, non_relevant_docs, number_new_terms):
        weights = self.weights
        relevant_docs_weights = self.get_doc_weights(relevant_docs)
        non_relevant_docs_weights = self.get_doc_weights(non_relevant_docs)

        # Compute weight vector with relevant and non relevant docs effect
        for term in relevant_docs_weights.keys():
            if term in non_relevant_docs_weights.keys():
                weights[term] = weights[term] + self.beta * (relevant_docs_weights[term]/len(relevant_docs)) - self.gamma * (non_relevant_docs_weights[term]/len(non_relevant_docs))
            else:
                weights[term] = weights[term] + self.beta * (relevant_docs_weights[term] / len(relevant_docs))

        # Compute new query vector
        filtered = {}
        for term in weights.keys():
            if term in self.query_vector.keys():
                self.new_query_vector[term] = self.alpha * self.query_vector[term] + weights[term]
                filtered.update({term: self.new_query_vector[term]})
            elif weights[term] > 0:
                self.new_terms[term] = weights[term]
                self.new_query_vector[term] = weights[term]

        index = 0
        for w in sorted(self.new_terms, key=self.new_terms.get, reverse=True):
            if index < number_new_terms: #e.g. add only 2 new terms into initial query vector
                filtered.update({w: self.new_terms[w]})
                index += 1
            else:
                break
        return filtered

    # return : additional terms vector based on the Rocchio
    def get_topk_new_terms(self, top_k):
        filtered = {}
        index = 0
        new_terms = self.new_terms
        for w in sorted(new_terms, key=new_terms.get, reverse=True):
            if index < top_k:
                filtered.update({w: new_terms[w]})
                index += 1
            else:
                break
        return filtered
