#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.model_selection import cross_validate, StratifiedKFold


class Classifier:
    classifiers = {
        "Naive Bayes" : GaussianNB(),
        "K Nerest Neighbors": KNeighborsClassifier(3),
        "Linear SVM": SVC(kernel="linear", C=0.025),
        "RBF SVM": SVC(gamma=2, C=1),
        "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Neural Net": MLPClassifier(alpha=1),
        "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        "AdaBoost": AdaBoostClassifier(),
        "xgboost": xgb.XGBClassifier()
        }

    @classmethod
    def evaluate_classification(cls, X, y):
        """cross-validation on a train set and return evaluation result
        :param X: data features
        :param y: data labels
        :return: metric_results (Accuracy, Precision, Recall, F1)
        """
        for name, clf in cls.classifiers.items():
            cv = StratifiedKFold(n_splits=5)
            scoring = {'ACC': 'accuracy', 'P': 'precision', 'R': 'recall', 'F1': 'f1'}
            results = cross_validate(clf, X, y=y, cv=cv, scoring=scoring)
            metric_results = [np.mean(results[idx]) for idx in ['test_ACC', 'test_P', 'test_R', 'test_F1']]
            print(f'{name}: A={metric_results[0]:.2f} P={metric_results[1]:.2f} R={metric_results[2]:.2f} F1={metric_results[3]:.2f}')
        return metric_results
