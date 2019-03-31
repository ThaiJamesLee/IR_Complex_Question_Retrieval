import math
__author__ = 'Duc Tai Ly'


class BM25:

    # documents: is a dict of doc_ids containing terms with corresponding weights
    # e.g. {'id_1': {'term1': value1, 'term2': value2, ... }, 'id_2': {'term12': value12, 'term2': value22, ... }, ...}
    # default: k = 1.2
    # default: b = 0.75
    def __init__(self, documents, b=None, k=None):
        self.documents = documents
        if b is None:
            self.b = 0.75
        if k is None:
            self.k = 1.2

    # documents average length
    def get_average_doc_length(self):
        num_docs = len(self.documents.keys())
        num_items = 0
        for k, v in self.documents.items():
            num_items = num_items + len(v)
        return num_items/num_docs

    # document exact length of specified doc_id
    def get_doc_length (self, doc_id):
        return len(self.documents[doc_id])

    # get frequency of term in document with doc_id
    def get_term_frequency_in_doc(self, term, doc_id):
        try:
            return self.documents[doc_id][term]
        except KeyError:
            return 0

    # get weight of term as idf weight
    def idf_weight(self, term):
        df = 0
        num_docs = len(self.documents.keys())
        for k, v in self.documents.items():
            if term in v:
                df = df + 1
        return math.log(num_docs/df, 10)

    # calculate relevance score of query and corresponding document
    # terms in query should only be separated by blank spaces
    def relevance(self, doc_id, query):
        score = 0
        terms = query.split()

        for term in terms:
            counter = self.get_term_frequency_in_doc(term, doc_id) * (self.k + 1)
            denominator = self.get_term_frequency_in_doc(term, doc_id) + (self.k * self.b * self.get_doc_length(doc_id)/self.get_average_doc_length()) + self.k * (1 - self.b)
            idf = self.idf_weight(term)
            score = score + (idf * counter/denominator)
        return score
