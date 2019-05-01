import numpy as np
import pandas as pd
import pickle
from tf_idf import TFIDF
from utils import Utils
from cache_query_and_docs_as_embedding_vectors import Caching


class createInputForL2R:
    """
    Prepare the input for L2R


    """
    def __init__(self, bm25, tfidf, glove, tfidf_rocchio, glove_rocchio):
        """
        train_x, train_y, train_q, test_x, test_y, test_q
        train_x, test_x have below structure for each query, doc pair
        cosine_similarity score of [tfidf, glove, tfidf_rocchio, glove_rocchio]
        :param tfidf: filename
        :param glove: filename
        :param tfidf_rocchio: filename
        :param glove_rocchio: filename
        """
        self.bm25_score = pickle.load(open(bm25, 'rb'))
        self.tfidf_score = pickle.load(open(tfidf, 'rb'))
        self.glove_score = pickle.load(open(glove, 'rb'))
        self.tfidf_rocchio_score = pickle.load(open(tfidf_rocchio, 'rb'))
        self.glove_rocchio_score = pickle.load(open(glove_rocchio, 'rb'))

        train = pickle.load(open('processed_data/process_train.pkl', 'rb'))
        test = pickle.load(open('processed_data/process_test.pkl', 'rb'))

        self.train_x, self.train_y, self.train_q = self.get_feature(train)
        self.test_x, self.test_y, self.test_q = self.get_feature(test)
        self.input_for_package(train, test)

    def get_feature(self, df):
        num = df.shape[0]
        x = np.zeros((num, 5))
        y = df['rel'].values
        qlist = df['query'].values
        for i in range(num):
            q = df['query'].values[i]
            d = df['docid'].values[i]
            x[i, 0] = Utils.get_value_from_key_in_dict(self.bm25_score, d, q)
            x[i, 1] = Utils.get_value_from_key_in_dict(self.tfidf_score, d, q)
            x[i, 2] = Utils.get_value_from_key_in_dict(self.glove_score, d, q)
            x[i, 3] = Utils.get_value_from_key_in_dict(self.tfidf_rocchio_score, d, q)
            x[i, 4] = Utils.get_value_from_key_in_dict(self.glove_rocchio_score, d, q)
        return x, y, qlist

    def input_for_package(self, train_df, test_df):
        ## train
        input_train = pd.DataFrame(train_df['rel'].values)
        input_train = input_train.assign(qid=train_df['query'].values, f1=self.train_x[:, 0], f2=self.train_x[:, 1],
                                       f3=self.train_x[:, 2], f4=self.train_x[:, 3], f5=self.train_x[:, 4],
                                       docid=train_df['docid'].values)
        with open("input_for_L2R/input_train.txt", "w") as text_file:
            for i in range(train_df.shape[0]):
                print(f"{input_train.iloc[i, 0]} qid:{input_train.iloc[i, 1]} "
                      f"1:{input_train.iloc[i, 2]} 2:{input_train.iloc[i, 3]} "
                      f"3:{input_train.iloc[i, 4]} 4:{input_train.iloc[i, 5]} "
                      f"5:{input_train.iloc[i, 6]} #docid{input_train.iloc[i, 7]} ", file=text_file)

        ## test
        input_test = pd.DataFrame(test_df['rel'].values)
        input_test = input_test.assign(qid=test_df['query'].values, f1=self.test_x[:,0], f2=self.test_x[:,1],
                                       f3=self.test_x[:,2], f4=self.test_x[:,3], f5=self.test_x[:,4],
                                       docid=test_df['docid'].values)

        # input_test.to_csv('input_for_L2R/input_test.csv', header=False, index=False)
        with open("input_for_L2R/input_test.txt", "w") as text_file:
            for i in range(train_df.shape[0]):
                print(f"{input_test.iloc[i,0]} qid:{input_test.iloc[i,1]} "
                      f"1:{input_test.iloc[i,2]} 2:{input_test.iloc[i,3]} "
                      f"3:{input_test.iloc[i,4]} 4:{input_test.iloc[i,5]} "
                      f"5:{input_test.iloc[i,6]} #docid{input_test.iloc[i,7]} ", file=text_file)


createInputForL2R('cache/bm25_scores.pkl', 'cache/cosine_tf_idf.pkl', 'cache/cosine_sem_we.pkl',
                  'cache/cosine_tf_idf.pkl', 'cache/cosine_sem_we_query_exp.pkl')