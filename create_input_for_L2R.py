import numpy as np
import pandas as pd
import pickle
from utils import Utils


class createInputForL2R:
    """
    Prepare the input for L2R (RankLib format)

    """

    def __init__(self, train, test):

        self.bm25_score = pickle.load(open('cache/bm25_scores.pkl', 'rb'))
        self.tfidf_score = pickle.load(open('cache/cosine_tf_idf.pkl', 'rb'))
        self.glove_score = pickle.load(open('cache/cosine_sem_we.pkl', 'rb'))
        self.tfidf_rocchio_score = pickle.load(open('cache/cosine_query_expansion.pkl', 'rb'))
        self.glove_rocchio_score = pickle.load(open('cache/cosine_sem_we_query_exp.pkl', 'rb'))

        train = pickle.load(open(train, 'rb'))
        test = pickle.load(open(test, 'rb'))
        createInputForL2R.input_for_package(train, 'train')
        createInputForL2R.input_for_package(test, 'test')

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

    @staticmethod
    def input_for_package(df, filename):
        x, _, query = createInputForL2R.get_feature(df)
        qid = np.zeros(df.shape[0])
        index = 1
        for i in range(len(qid) - 1):
            qid[i] = index
            if query[i] != query[i + 1]:
                index += 1
        qid[len(qid) - 1] = index
        qid = qid.astype(int)

        input = pd.DataFrame(df['rel'].values)
        input = input.assign(qid=qid, f1= x[:, 0], f2=x[:, 1], f3=x[:, 2], f4=x[:, 3], f5=x[:, 4],
                             docid=df['docid'].values)

        # input_test.to_csv('input_for_L2R/input_test.csv', header=False, index=False)
        with open(f"input_for_L2R/input_{filename}.txt", "w") as text_file:
            for i in range(df.shape[0]):
                print(f"{input.iloc[i,0]} qid:{input.iloc[i,1]} "
                      f"1:{input.iloc[i,2]} 2:{input.iloc[i,3]} "
                      f"3:{input.iloc[i,4]} 4:{input.iloc[i,5]} "
                      f"5:{input.iloc[i,6]} #docid = {input.iloc[i,7]} ", file=text_file)


# createInputForL2R('process_data/process_train.pkl', 'process_data/process_test.pkl')