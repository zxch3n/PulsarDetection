import numpy as np
import model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, \
    roc_auc_score, roc_curve, f1_score, classification_report, recall_score
from sklearn.model_selection import cross_val_score, ShuffleSplit


def cross_validation(cls, X, y, scoring='f1', n_jobs=-1, n_splits=3):
    """

    :param cls: the classifier, inhered from BaseModel
    :param X: input features
    :param y: input class
    :param scoring: {'f1', 'roc_auc', 'recall', 'all'}
    :param n_jobs: The
    :return: scores
    """
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)

    if scoring != 'all':
        return cross_val_score(cls, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)

    # This faster implementation is not allowed...
    # Because it can only allow 1 number output
    # return cross_val_score(cls, X, y, cv=cv, scoring=_scoring_func, n_jobs=n_jobs)

    # Thus I choose a stupid one
    output = {}
    for scoring in ('f1', 'roc_auc', 'recall'):
        output[scoring] = cross_val_score(cls, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    return output


def estimate(cls, X_train, X_test, y_train, y_test, confusion_matrix=False):
    if not isinstance(cls, model.BaseModel):
        model.BaseModel.register(type(cls))

    cls.fit(X_train, y_train)

    y_score_test = cls.predict_proba(X_test)
    y_pred_test = cls.predict(X_test)

    y_score_train = cls.predict_proba(X_train)
    y_pred_train = cls.predict(X_train)

    assert y_score_test.dtype.kind == 'f',\
        "Predict Proba value should be float. Change the predict definition in {}".format(cls.__class__)
    assert np.all(np.equal(y_pred_test, y_pred_test.astype(int))), \
        "Predict value should be int. Change the predict definition in {}".format(cls.__class__)

    if confusion_matrix:
        plot_confusion_matrix(y_true=y_test, y_pred=y_pred_test)

    y_score_test = score_transform(y_score_test)
    y_score_train = score_transform(y_score_train)
    return {
        'train': {
            'roc_auc': roc_auc_score(y_true=y_train, y_score=y_score_train),
            'f1': f1_score(y_true=y_train, y_pred=y_pred_train)
        },
        'test': {
            'roc_auc': roc_auc_score(y_true=y_test, y_score=y_score_test),
            'f1': f1_score(y_true=y_test, y_pred=y_pred_test)
        }
    }


def score_transform(y_score):
    if len(y_score.shape) == 1:
        return y_score
    if y_score.shape[1] == 1:
        return y_score[:, 0]
    assert y_score.shape[1] == 2 and len(y_score.shape) == 2, "Not support y_score type with shape {}".format(y_score)
    return y_score[:, 1]


def plot_confusion_matrix(y_pred, y_true):
    cnf_matrix = confusion_matrix(y_pred, y_true)
    print("the recall for this model is :", cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[1, 0]))
    sns.heatmap(cnf_matrix, cmap="coolwarm_r", annot=True, linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    print("\n----------Classification Report------------------------------------")
    print(classification_report(y_true, y_pred))


def _scoring_func(estimator, X, y):
    pred = estimator.predict(X)
    score = estimator.predict_proba(X)
    return [
        f1_score(y_true=y, y_pred=pred),
        recall_score(y_true=y, y_pred=pred),
        roc_auc_score(y_true=y, y_score=score)
    ]


__all__ = ['cross_validation']
