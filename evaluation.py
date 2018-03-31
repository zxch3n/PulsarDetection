import numpy as np
import model
import pandas as pd
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, \
    roc_auc_score, roc_curve, f1_score, classification_report, recall_score
from sklearn.model_selection import cross_val_score, ShuffleSplit, RandomizedSearchCV, \
    GridSearchCV


def cross_validation(cls, X, y, scoring='f1', n_jobs=-1, n_splits=3):
    """

    :param cls: the classifier, inhered from BaseModel
    :param X: input features
    :param y: input class
    :param scoring: {'f1', 'roc_auc', 'both', 'all'}
    :param n_jobs: The
    :return: scores
    """
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)

    # This faster implementation is not allowed...
    # Because it can only allow 1 number output
    # return cross_val_score(cls, X, y, cv=cv, scoring=_scoring_func, n_jobs=n_jobs)

    # Thus I choose a stupid one
    if scoring == 'both':
        output = {}
        for scoring in ('f1', 'roc_auc'):
            output[scoring] = cross_val_score(cls, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        return output
    if scoring == 'all':
        output = {}
        for scoring in ('f1', 'roc_auc', 'precision', 'recall'):
            output[scoring] = cross_val_score(cls, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        return output
    if scoring != 'both' and scoring != 'all':
        return cross_val_score(cls, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)


def estimate(cls, X_train, X_test, y_train, y_test, use_confusion_matrix=False):
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

    if use_confusion_matrix:
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


def best_param_search(estimator, params, X, y, verbose=True, n_jobs=-1):
    """
    The automatic search method

    :param estimator: the learner
    :param params:
        A list of param list. the search will start tuning from
        the first 1.

        For example:
            [
                {'C': [0.01, 0.1, 1], 'kernel':['rbf', 'poly']},
                {'gamma': [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]}
            ]
        This method will grid search `C` and `kernel` params first,
        by cross validation, using the default `gamma` value.
        And then use the best `C` and `kernel` params to grid search
        the best setting of `gamma`, so on and so forth.
    :param X: features
    :param y: labels
    :param verbose: {True, False} whether print the info while tuning
    :return:
        best_params: dict. {'C': 0.1, 'kernel': 'rbf', 'gamma': 0.1}
        df_scores: pd.DataFrame(index=params, columns=k_fold_score)
    """
    best_params = {}
    df_scores = pd.DataFrame(columns=['test_score', 'train_score', 'fit_time', 'score_time'])
    _estimator = estimator
    for ps in params:
        estimator = clone(_estimator)
        for name, value in best_params.items():
            ps[name] = [value]
        cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
        clf = GridSearchCV(estimator, ps, scoring='f1', cv=cv, n_jobs=n_jobs, return_train_score=True)
        clf.fit(X, y)
        for name, value in clf.best_params_.items():
            best_params[name] = value

        for i, dikt in enumerate(clf.cv_results_['params']):
            index_name = ';'.join(['{}:{}'.format(a, b) for a, b in dikt.items()])
            df_scores.loc[index_name] = [
                clf.cv_results_['mean_test_score'][i],
                clf.cv_results_['mean_train_score'][i],
                clf.cv_results_['mean_fit_time'][i],
                clf.cv_results_['mean_score_time'][i],
            ]
    return best_params, df_scores


__all__ = ['cross_validation']
