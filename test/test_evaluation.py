import model
import numpy as np
import evaluation
import preprocess
from sklearn.model_selection import train_test_split

X, y = preprocess.load_data('./HTRU2/HTRU_2.csv')
X, y = X[:100], np.array([1]*20 + [0]*80)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)


def test_cross_validation():
    for md in [
        model.SVM(),
        model.MultiClassesLearner('KNN', {'n_neighbors': 1}),
        model.KNN(),
        model.XGBoost(),
        model.LinearModel(),
        model.DecisionTree(),
    ]:
        try:
            evaluation.cross_validation(md, X, y, scoring='both', n_jobs=1, n_splits=2)
        except Exception as e:
            print(md.__class__)
            raise e


def test_estimate():
    for md in [
        model.SVM(),
        model.KNN(),
        model.XGBoost(),
        model.LinearModel(),
        model.DecisionTree()
    ]:
        evaluation.estimate(md, X_train, X_test, y_train, y_test)


def test_data_split_identical():
    X_, y_ = X[:40], np.array([0] * 20 + [1] * 20)
    for md in [
        model.SVM(),
        model.KNN(),
        model.XGBoost(),
        model.LinearModel(),
        model.DecisionTree()
    ]:
        a = evaluation.estimate(md, X_train, X_test, y_train, y_test)
        b = evaluation.estimate(md, X_train, X_test, y_train, y_test)
        assert a == b
        a = evaluation.cross_validation(md, X_, y_, scoring='both', n_splits=2, n_jobs=1)
        b = evaluation.cross_validation(md, X_, y_, scoring='both', n_splits=2, n_jobs=1)
        assert np.all(a['f1'] == b['f1'])
        assert np.all(a['roc_auc'] == b['roc_auc'])
