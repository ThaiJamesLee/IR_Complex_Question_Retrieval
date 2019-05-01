# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'

import pandas as pd
import csv


class WordEmbedding:
    """
    Opens the pre-trained glove model.
    """
    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
        self.model = self.load_glove_model()

    def load_glove_model(self):
        """
        Implementation from stackoverflow. Accessed on 12.04.2019.
        https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
        :return: Returns the model as a dict.
        """
        if self.path_to_model is not None:
            print('Loading pre-trained glove model.')
            model = pd.read_csv(self.path_to_model, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
            return model
        else:
            raise Exception('No path given for pre-trained embedding model. It should be a .txt file!')

    def get_word_vector_2(self, word):
        return self.model.loc[word].values
