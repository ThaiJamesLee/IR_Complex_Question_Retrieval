__author__ = 'Duc Tai Ly'
import math


class Similarity:

    # calculate the cosie similarity score between two vectors v1, v2
    # We assume the vectors are dicts of key, value pairs
    # Like this: v1 = {k1:val1, k2:val2, k3:val3}
    @staticmethod
    def cosine_similarity(vec1, vec2):
        print('Compare:')
        print(vec1)
        print(vec2)
        counter = 0
        denominator_1 = 0
        denominator_2 = 0
        for k1, v1 in vec1.items():
            denominator_1 += math.pow(v1, 2)
            try:
                counter += v1 * vec2[k1]
            except KeyError:
                continue
        for k2, v2 in vec2.items():
            denominator_2 += math.pow(v2, 2)

        denominator_1 = math.sqrt(denominator_1)
        denominator_2 = math.sqrt(denominator_2)
        score = 0
        if denominator_1 != 0 or denominator_2 != 0:
            score = counter/(denominator_1*denominator_2)
        return score
