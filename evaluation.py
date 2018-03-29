import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, \
    roc_auc_score, roc_curve, f1_score, classification_report
from sklearn.model_selection import cross_val_score


def cross_validation(learner, X, y, scoring='f1'):
    """

    :param learner:
    :param X:
    :param y:
    :param scoring: {'f1', 'roc_auc'}
    :return:
    """
    return cross_val_score(learner, X, y, cv=3, scoring=scoring, n_jobs=-1)


def estimate(model, features_train, features_test, labels_train, labels_test):
    clf = model
    clf._fit(features_train, labels_train.values.ravel())
    pred = clf._predict(features_test)
    return roc_auc_score(labels_test, pred), f1_score(y_true=labels_test, y_pred=pred)


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
