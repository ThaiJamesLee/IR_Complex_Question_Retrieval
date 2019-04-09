

class RocchioOptimizeQuery:
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

    def __init__(self, initial_query_vector, tf_idf_matrix, alpha=1, beta=0.75, gamma=0.15):
        self.query_vector = initial_query_vector
        self.new_query_vector = {}
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.matrix = tf_idf_matrix

    # e.g. list of documents that bm25 scores more than 0.1 are considered relevant
    def get_relevant_docs(self, dic):
        # use Class Performance filter function..
        # filter_relevance_by_threshold, filter_relevance_by_top_k
        rel_docs = {}
        return rel_docs

    # e.g. list of documents that bm25 scores less than 0.1 are considered no relevant
    def get_nonrelvevant_docs(self, dic):
        # use filter_pred_negative
        non_reldocs={}
        return non_reldocs

    # document_list : a dict of doc_ids with its terms
    # e.g. {doc_id1 : [term1, term2 ..], doc_id2 : [term1, term2, ..]...}
    # relevant_docs : a list of doc_ids that are relevant
    # non_relevant_docs : a list of doc_ids that are non relevant
    # output new query vector
    def execute_rocchio(self, relevant_docs, non_relevant_docs):
        matrix = self.matrix
        # initialize weight vector for each term across all documents
        weights = {}
        for doc_id in matrix.keys():
            doc = matrix[str(doc_id)]
            for term in list(doc.keys()):
                weights[term] = 0.0

        # Sum the set of vectors of relevant documents
        relevant_docs_weights = {}
        for doc_id in relevant_docs:
            doc = matrix[str(doc_id)]
            for term in list(doc.keys()):
                if term in relevant_docs_weights:
                    relevant_docs_weights[term] = relevant_docs_weights[term] + doc[term]
                else:
                    relevant_docs_weights[term] = doc[term]

        # Sum the set of vectors of non relevant documents
        non_relevant_docs_weights = {}
        for doc_id in non_relevant_docs:
            doc = matrix[str(doc_id)]
            for term in list(doc.keys()):
                if term in non_relevant_docs_weights:
                    non_relevant_docs_weights[term] = non_relevant_docs_weights[term] + doc[term]
                else:
                    non_relevant_docs_weights[term] = doc[term]

        # Compute weight vector with relevant and non relevant docs effect
        for term in weights.keys():
            if term in relevant_docs_weights.keys():
                if term in non_relevant_docs_weights.keys():
                    weights[term] = weights[term] + self.beta * (relevant_docs_weights[term]/len(relevant_docs)) - self.gamma * (non_relevant_docs_weights[term]/len(non_relevant_docs))
                else:
                    weights[term] = weights[term] + self.beta * (relevant_docs_weights[term]/len(relevant_docs))
            else:
                weights[term] = weights[term] - self.gamma * (non_relevant_docs_weights[term]/len(non_relevant_docs))

        # Compute new query vector, *to do : save the positive weight for getting new terms
        for term in weights.keys():
            if term in self.query_vector.keys():
                self.new_query_vector[term] = self.alpha * self.query_vector[term] + weights[term]
            elif weights[term] > 0:
                self.new_query_vector[term] = weights[term]

        return self.new_query_vector

    # additional terms added based on the Rocchio
    def get_new_terms(self):
        pass
