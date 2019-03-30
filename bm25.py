import math

class BM25:
    b = 0.75
    k = 1.2

    # documents: is a dict of doc_ids containing terms with corresponding weights
    def __init__(self, documents, query):
        self.documents = documents
        self.query = query

    # documents average length
    def get_average_doc_length (self):
        num_docs = len(self.documents.keys())
        num_items = 0
        for k, v in self.documents.items():
            num_items = num_items + len(v)
        return num_items/num_docs

    # document exact length
    def get_doc_length (self, doc_id):
        return len(self.documents[doc_id])

    # get frequency of term in document
    def get_term_frequency_in_doc (self, term, doc_id):
        try:
            return self.documents[doc_id][term]
        except KeyError:
            return 0

    # get weight of term
    def idf_weight(self, term):
        df = 0
        num_docs = len(self.documents.keys())
        for k, v in self.documents.items():
            if term in v:
                df = df + 1
        return math.log(num_docs/df)

    # calculate relevance score of query and corresponding document
    # terms in query should only be separated by blank spaces
    def relevance(self, doc_id, query):
        score = 0
        terms = query.split()

        for term in terms:
            counter = self.get_term_frequency_in_doc(term, doc_id) * (BM25.k + 1)
            denominator = self.get_term_frequency_in_doc(term, doc_id) + (BM25.k * BM25.b * self.get_doc_length(doc_id)/self.get_average_doc_length()) + BM25.k * (1 - BM25.b)
            idf = self.idf_weight(term)
            score = score + (idf * counter/denominator)
        return score