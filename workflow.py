# 1. Load data or run data preprocess, if use existing -> load data, if new dataset, then run preprocess
# --> dataframe for test and train, run doc_structure function for input bm25
# 2. calculate bm25 --> candidates
# --> dict {query: {docid: score, ...}, ...}
# --> dump as object, use pickle
# 3. format to tf-idf
# --> dict {docid: {term: value, ...}, ...}
# 4. query expansion with tf-idf
# 5. merge results of bm25, tf-idf (cosine), tf-idf+query expansion (cosine) --> new dataframe
# 6. put into L2R and Duet Models
# 7. evaluate (bm25, tf_idf(+query expansion), L2R, Duet) using MAP, R-Prec, MRR
# 8. create new query and visualize result

from preprocessing import Preprocess
from cache_embedding_vectors import Caching
from create_features import FeatureGenerator
from tf_idf import TFIDF
import create_input_for_L2R
from metrics_calculate import Metrics
import os

import cache_word_embeddings

process_type = 'lemma'

# file paths to the test200 data
train_paragraphs = 'test200-train/train.pages.cbor-paragraphs.cbor'
train_pages = 'test200-train//train.pages.cbor'
train_pages_toplevel = 'test200-train/train.pages.cbor-toplevel.qrels'
train_pages_hierarchical = 'test200-train/train.pages.cbor-hierarchical.qrels'
train_pages_article = 'test200-train/train.pages.cbor-article.qrels'


# run metrics variables

# filter scores by threshold
threshold = 0.0

# true: use only documents that have the true label for metrics calculation
only_actual = False

# run metrics calculation with multiple threads
exec_with_multithread = True


def execute():
    Preprocess(process_type, train_paragraphs, train_pages, train_pages_toplevel, train_pages_hierarchical,
               train_pages_article)

    # execute this to cache semantic word embeddings
    # this requires the glove.840B.300d.txt file in this directory

    # cache_word_embeddings.cache_terms_embedding_vectors()

    c = Caching(process_type=process_type)

    tf_idf = TFIDF(c.doc_structure).term_doc_matrix

    c.create_document_embeddings(tf_idf)
    c.create_query_embeddings(tf_idf)

    feature_generator = FeatureGenerator(caching=c, tf_idf=tf_idf)

    feature_generator.calculate_bm25()
    feature_generator.calculate_cosine_tf_idf()

    # this requires the bm25 scores cached
    c.create_query_embeddings_query_expansion(tf_idf)

    feature_generator.calculate_cosine_tf_idf_query_expansion()
    feature_generator.calculate_cosine_semantic_embeddings_query_expansion()
    feature_generator.calculate_cosine_semantic_embeddings()

    create_input_for_L2R.createInputForL2R('process_data/process_train.pkl', 'process_data/process_test.pkl')

    # calculate metrices for bm25, tf-idf, tf-idf + rocchio, glove, glove + rocchio
    m = Metrics()
    if exec_with_multithread:
        m.excecute_multithreaded(threshold=threshold, only_actual=only_actual)
    else:
        m.execute_singethreaded(threshold=threshold, only_actual=only_actual)

    # run L2R
    # TODO


if __name__ == "__main__":
    execute()



