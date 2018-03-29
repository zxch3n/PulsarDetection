import pandas as pd
import numpy as np
import time
import random


def normalize(X):
    return X


def train_test_split(X, y, test_set_rate=0.3, random_state=123, slice_index=0):
    """

    TODO make sure to keep the same distribution

    :param X:
    :param y:
    :param test_set_rate
    :param random_state:
        The random seed feed to the random method
        If None, the random_state will be set to current time.
    :param slice_index:
        (int) Indicate which slice to pick.
        slice_index < 1 / test_set_rate
        It's useful when cross validating your model.
    :return: X_train, X_test, y_train, y_test
    """
    if random_state is None:
        random_state = int(time.time())
    random.seed(random_state)
    pass


def make_data_balanced(X, y, method='SMOTE', pos_neg_rate=1):
    """
    Make the classes balanced.

    THINK: Which way is better, use this before or after train_test_split?

    :param X
    :param y
    :param method: {'smote', 'oversampling', 'undersampling'}
    :param pos_neg_rate: The rate of positive samples to negative samples
    :return:
    """
    pass


def balance_data_by_creating_classes(X, y):
    """
    Let N = positive_num / negative_num
    Divide the negative samples to N different parts based on clusters.

    :param X
    :param y
    :return: [X_neg_1, X_neg_2, ..., X_neg_n], X_pos
        neg = negative samples
        pos = negative samples
    """
    pass

