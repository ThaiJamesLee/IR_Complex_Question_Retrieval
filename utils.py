# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'


# Utils class for helping functions
# like static functions that can be used globally
class Utils:

    # Helper to add nested dicts
    # add dict into a parent dict
    # matrix: the parent dict consist of docs as keys
    # key: value: child dict
    @staticmethod
    def add_ele_to_matrix(matrix, doc, key, value):
        if doc not in matrix:
            matrix.update({doc: {key: value}})
        else:
            matrix[doc].update({key: value})
        return matrix
