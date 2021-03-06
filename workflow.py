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
from metrics_calculate import Standard
from L2R import L2R_test
import os

import cache_word_embeddings

process_type = 'lemma'

# file paths to the test200 data
train_paragraphs = 'test200-train/train.pages.cbor-paragraphs.cbor'
train_pages = 'test200-train//train.pages.cbor'
train_pages_hierarchical = 'test200-train/train.pages.cbor-hierarchical.qrels'

# paths of files generated via Preprocess

# run metrics variables

# filter scores by threshold
threshold = 0.0
top_k = 10

# when using rocchio: expand query by the number of new terms
rocchio_terms = 5

# true: use only documents that have the true label for metrics calculation
# otherwise it takes the top_k relevant documents (according to the algorithms)
only_actual = True

# run metrics calculation with multiple threads
exec_with_multithread = True


# pipeline to run the learning to rank task
def execute_L2R_task():
    # Preprocess(process_type, train_paragraphs, train_pages, train_pages_hierarchical)

    # execute this to cache semantic word embeddings
    # this requires the glove.840B.300d.txt file in this directory
    # uncomment this if you have
    #
    # cache_word_embeddings.cache_terms_embedding_vectors()

    c = Caching(process_type=process_type)

    tf_idf = TFIDF(c.doc_structure)

    # save glove vectors for query and docs in cache
    c.create_document_embeddings(tf_idf.term_doc_matrix)
    c.create_query_embeddings(tf_idf)

    feature_generator = FeatureGenerator(caching=c, tf_idf=tf_idf)

    # calculate relevance scores for bm25 and tf-idf
    feature_generator.calculate_bm25()
    feature_generator.calculate_cosine_tf_idf()

    #
    # this requires the bm25 scores cached
    # save glove vectors for query + rocchio in cache
    c.create_query_embeddings_query_expansion(tf_idf, rocchio_terms=rocchio_terms)

    # calculate scores for tf-idf+rocchtio, glove+rocchtio, glove
    feature_generator.calculate_cosine_tf_idf_rocchio(rocchio_terms=rocchio_terms)
    feature_generator.calculate_cosine_glove_and_rocchio(rocchio_terms=rocchio_terms)
    feature_generator.calculate_cosine_glove()


    # calculate MAP, MRR, metrics for bm25, tf-idf, tf-idf + rocchio, glove, glove + rocchio
    # R-Prec not tested!
    m = Metrics(top_k=top_k)
    if exec_with_multithread:
        m.excecute_multithreaded(threshold=threshold, only_actual=only_actual)
    else:
        m.execute_singethreaded(threshold=threshold, only_actual=only_actual)

    # execute L2R
    create_input_for_L2R.createInputForL2R('process_data/process_train.pkl', 'process_data/process_test.pkl')
    # change L2R model with L2R_test.*_md e.g. AdaRank_md, ...
    L2R_test.execute_L2R(L2R_test.CoordinateAscent_md)


# create synthetic page
def execute_complex_answer_retrieval_task():
    # initialize
    c = Caching(process_type=process_type)
    tf_idf = TFIDF(c.doc_structure)
    feature_generator = FeatureGenerator(caching=c, tf_idf=tf_idf)

    # compute paragraph - paragraph scores and cache
    feature_generator.generate_bm25_doc_doc(b=0.75, k=1.2)

    feature_generator.generate_cosine_tfidf_doc_doc()
    feature_generator.generate_cosine_tfidf_rocchio_doc_doc(top_k=top_k, rocchio_terms=rocchio_terms)

    feature_generator.generate_cosine_glove_doc_doc()
    feature_generator.generate_cosine_glove_rocchio_doc_doc(top_k=top_k, rocchio_terms=rocchio_terms)

    m = Standard(only_actual=True)
    print(m.excecute_stand_multithreaded(threshold=0))
    print(m.excecute_avg_stand_multithreaded())


if __name__ == "__main__":
    # each task might take a lot of time
    # you might want to comment out the task you need
    execute_L2R_task()
    execute_complex_answer_retrieval_task()




