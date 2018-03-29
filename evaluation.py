import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, f1_score
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

    # print("the recall for this model is :", cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[1, 0]))
    # fig = plt.figure(figsize=(6, 3))  # to plot the graph
    # print("TP", cnf_matrix[1, 1,])  # no of fraud transaction which are predicted fraud
    # print("TN", cnf_matrix[0, 0])  # no. of normal transaction which are predited normal
    # print("FP", cnf_matrix[0, 1])  # no of normal transaction which are predicted fraud
    # print("FN", cnf_matrix[1, 0])  # no of fraud Transaction which are predicted normal
    # sns.heatmap(cnf_matrix, cmap="coolwarm_r", annot=True, linewidths=0.5)
    # plt.title("Confusion_matrix")
    # plt.xlabel("Predicted_class")
    # plt.ylabel("Real class")
    # plt.show()
    # print("\n----------Classification Report------------------------------------")
    # print(classification_report(labels_test, pred))


__all__ = ['cross_validation']
