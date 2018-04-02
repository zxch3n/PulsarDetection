import pandas as pd
import os
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
import random


def load_data(path='./HTRU2/HTRU_2.csv'):
    if not os.path.exists(path):
        path = os.path.join('../', path)

    df = pd.read_csv(path, header=None)
    df['Class'] = df.pop(8)
    y = df['Class']
    X = df.drop('Class', axis=1)
    return X, y


def load_train_test():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    return X_train, X_test, y_train, y_test


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


def upsampling(X, y, ratio=1.0, random_state=123):
    pos_num = sum(y)
    target_num = int(0.5 + ratio * (len(y) - pos_num))
    if not pos_num < target_num:
        return X, y
    X_pos = X[y == 1]
    X_pos = np.concatenate([X_pos for _ in range(target_num // pos_num)] + [X_pos[:target_num % pos_num]])
    X_neg = X[y == 0]
    X, y = np.concatenate([X_pos, X_neg]), np.array([1] * len(X_pos) + [0] * len(X_neg))
    return shuffle(X, y, random_state=random_state)


def downsampling(X, y, random_state=123, ratio=1.0):
    """

    :param X:
    :param y:
    :param random_state:
    :param ratio: pos to neg ratio
    :return:
    """
    random.seed(random_state)
    pos_num = sum(y)
    if not pos_num < ratio * (len(y) - pos_num):
        return X, y
    if isinstance(X, pd.DataFrame):
        X = X.values
    X_neg = X[y == 0]
    X_neg = X_neg[:int(pos_num / ratio)]
    X_pos = X[y == 1]
    X = np.concatenate([X_neg, X_pos])
    y = np.array([0] * len(X_neg) + [1] * pos_num)
    return shuffle(X, y)


# SMOTE


def smote_regular(X, y, ratio=1.0):
    s = SMOTE(random_state=123, n_jobs=-1, ratio=ratio, kind='regular')
    return s.fit_sample(X, y)


def smote_borderline1(X, y, ratio=1.0):
    s = SMOTE(random_state=123, n_jobs=-1, ratio=ratio, kind='borderline1')
    return s.fit_sample(X, y)


def smote_borderline2(X, y, ratio=1.0):
    s = SMOTE(random_state=123, n_jobs=-1, ratio=ratio, kind='borderline2')
    return s.fit_sample(X, y)


def smote_svm(X, y, ratio=1.0):
    s = SMOTE(random_state=123, n_jobs=-1, ratio=ratio, kind='svm')
    return s.fit_sample(X, y)


def get_smote(kind='regular'):
    """

    :param random_state:
    :param ratio:
    :param kind:
        ``'regular'``, ``'borderline1'``, ``'borderline2'``, ``'svm'``.
    :return:
    """
    dikt = {
        'regular': smote_regular,
        'borderline1': smote_borderline1,
        'borderline2': smote_borderline2,
        'svm': smote_svm
    }
    return dikt[kind]
