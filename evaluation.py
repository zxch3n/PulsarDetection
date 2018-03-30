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


def estimate(model, X_train, X_test, y_train, y_test):
    clf = model
    clf.fit(X_train, y_train.values.ravel())
    pred_score = clf.predict(X_test)
    assert pred_score.dtype.kind == 'f' and np.not_equal(pred_score, pred_score.astype(int)), \
        "Predict value should be float. Change the predict definition in {}".format(model.__class__)
    y_pred = _score_to_pred(pred_score, y_train)
    return roc_auc_score(y_test, pred_score), f1_score(y_true=y_test, y_pred=y_pred)


def _score_to_pred(scores, y_train):
    threshold = _get_best_threshold(scores, y_train)
    return np.array([[1] if x > threshold else [0] for x in scores])


def _get_best_threshold(scores, y_train):
    """

    :param scores:
    :param y_train:
    :return: threshold.
        pred = 1, if score > threshold;
        pred = 0, otherwise;
    """
    combined = [x for x in zip(scores, y_train)]
    combined.sort(key=lambda x: x[0])
    threshold = combined[0][0] - 0.1
    pos_len = np.sum(y_train)
    tp, fp, fn = pos_len, len(y_train) - pos_len, 0
    f1 = tp / (fp + fn)

    best_f1, best_threshold = f1, threshold
    for threshold, y in combined:
        if y == 1:
            tp -= 1
            fn += 1
        else:
            fp -= 1
        f1 = tp / (fp + fn)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    return best_threshold


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
