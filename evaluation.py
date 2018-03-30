import numpy as np
import model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, \
    roc_auc_score, roc_curve, f1_score, classification_report
from sklearn.model_selection import cross_val_score, ShuffleSplit


def cross_validation(learner, X, y, scoring='f1'):
    """

    :param learner:
    :param X:
    :param y:
    :param scoring: {'f1', 'roc_auc'}
    :return:
    """
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    return cross_val_score(learner, X, y, cv=cv, scoring=scoring, n_jobs=-1)


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


__all__ = ['cross_validation']
