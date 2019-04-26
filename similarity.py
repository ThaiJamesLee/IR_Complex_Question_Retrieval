# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'
import math


class Similarity:

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        calculate the cosine similarity score between two vectors vec1, vec2
        We assume the vectors are dicts of key, value pairs
        :param vec1: a dictionary {k1:val1, k2:val2, k3:val3}, where value are numerics
        :param vec2: a dictionary {k1:val1, k2:val2, k3:val3}, where value are numerics
        :return: Cosine Similarity score
        """
        counter = 0
        denominator_1 = 0
        denominator_2 = 0
        for k1, v1 in vec1.items():
            denominator_1 += math.pow(v1, 2)
            try:
                counter += v1 * vec2[k1]
            except KeyError:
                pass
        for k2, v2 in vec2.items():
            denominator_2 += math.pow(v2, 2)

        denominator_1 = math.sqrt(denominator_1)
        denominator_2 = math.sqrt(denominator_2)
        score = 0
        if denominator_1 != 0 and denominator_2 != 0:
            score = counter/(denominator_1*denominator_2)
        return score

    @staticmethod
    def cosine_similarity_normalized(vec1, vec2):
        """
        calculate the cosine similarity score between two vectors vec1, vec2
        We assume the vectors are dicts of key, value pairs.
        We assume the vectors are normalized vectors, thus, we skip the denominator calculation
        :param vec1: a dictionary {k1:val1, k2:val2, k3:val3}, where value are numerics
        :param vec2: a dictionary {k1:val1, k2:val2, k3:val3}, where value are numerics
        :return: Cosine Similarity score
        """
        counter = 0
        for k1, v1 in vec1.items():
            try:
                counter += v1 * vec2[k1]
            except KeyError:
                pass
        return counter
