import numpy as np
import xgboost as xgb
from abc import ABCMeta, abstractmethod
from six import with_metaclass
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import ClassifierMixin

normalizers = {'minmax': MinMaxScaler, 'standard': StandardScaler}


class BaseModel(with_metaclass(ABCMeta, ClassifierMixin)):
    @abstractmethod
    def __init__(self, normalizer=None, imbalance_method=None):
        """

        :param normalizer:
        :param imbalance_method:
        """
        if normalizer in normalizers:
            self.normalizer = normalizers[normalizer]()
        else:
            self.normalizer = None
        self.imbalance_method = imbalance_method

    @abstractmethod
    def _predict(self, X):
        pass

    @abstractmethod
    def _predict_proba(self, X):
        pass

    @abstractmethod
    def _fit(self, X, y):
        pass

    def fit(self, X, y):
        if self.normalizer is not None:
            self._normalize_fit(X)
            X = self._normalize_transform(X)

        # imbalance method may use KNN/SVM to generate data
        # so it should be used after the normalization process
        if self.imbalance_method is not None:
            X, y = self.imbalance_method(X, y)

        self._fit(X, y)

    def predict(self, X):
        X = self._normalize_transform(X)
        return self._predict(X)

    def predict_proba(self, X):
        X = self._normalize_transform(X)
        return self._predict_proba(X)

    def set_sample_method(self, method):
        self.imbalance_method = method
        return self

    def _normalize_fit(self, X):
        self.normalizer.fit(X)

    def _normalize_transform(self, X):
        if self.normalizer is None:
            return X
        return self.normalizer.transform(X)


class BaseEnsembleModel(with_metaclass(ABCMeta, BaseModel)):
    @abstractmethod
    def __init__(self, learners):
        super(BaseEnsembleModel, self).__init__('')  # do not need normalization
        for learner in learners:
            assert isinstance(learner, BaseModel)
        self.learners = learners


class VotingEnsemble(BaseEnsembleModel):
    def __init__(self, learners, weights=None):
        super(VotingEnsemble, self).__init__(learners)
        self.classifier = VotingClassifier(
            estimators=[('{}_{}'.format(learner.__class__, i), learner) for i, learner in enumerate(learners)],
            voting='soft',
            weights=weights
        )

    def _fit(self, X, y):
        self.classifier.fit(X, y)

    def _predict(self, X):
        return self.classifier.predict(X)

    def _predict_proba(self, X):
        return self.classifier.predict_proba(X)


class LinearEnsemble(BaseEnsembleModel):
    def __init__(self, learners, validation_rate=0.4):
        super(LinearEnsemble, self).__init__(learners)
        self.weights = np.ones(len(learners))
        self.learners = learners
        self.lr_learner = LinearRegression(normalize=True)
        self.validation_rate = validation_rate
        self._threshold = 0

    def _fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=123, test_size=self.validation_rate)
        for learner in self.learners:
            learner.fit(X_train, y_train)
        ys = self._raw_predict(X_val)
        self.lr_learner.fit(ys, y_val)
        self._set_threshold(y_val, self.lr_learner.predict(ys))

    def _raw_predict(self, X):
        """

        :param X: the input X from outside
        :return: a concatenated matrix contains all the predictions from every sub-learner
        """
        ys = np.array([ln.predict_proba(X) for ln in self.learners])
        return ys.T

    def _predict(self, X):
        return np.asarray(self.lr_learner.predict(self._raw_predict(X)) > self._threshold, np.int8)

    def _predict_proba(self, X):
        return self.lr_learner.predict(self._raw_predict(X))

    def _set_threshold(self, y_true, y_pred_prob):
        self._threshold = _get_best_threshold(y_true=y_true, y_pred_prob=y_pred_prob)


class XGBoost(BaseModel):
    def __init__(self, balanced_learning=True, normalizer='minmax'):
        super(XGBoost, self).__init__(normalizer)
        self.xgb = xgb.XGBClassifier(n_jobs=-1)
        self.balanced = balanced_learning

    def _predict(self, X):
        return self.xgb.predict(X)

    def _predict_proba(self, X):
        proba = self.xgb.predict_proba(X)
        return proba[:, 1]

    def _fit(self, X, y):
        pos_num = np.sum(y)
        neg_num = len(y) - pos_num
        if self.balanced:
            self.xgb.scale_pos_weight = neg_num / pos_num
        self.xgb.fit(X, y)


class DecisionTree(BaseModel):
    def __init__(self, balanced_learning=True, max_depth=None, ):
        super(DecisionTree, self).__init__('', None)
        if balanced_learning:
            class_weight = 'balanced'
        else:
            class_weight = None
        self.tree = DecisionTreeClassifier(
            class_weight=class_weight,
            max_depth=max_depth,
            min_samples_leaf=5,
            random_state=123
        )

    def _predict(self, X):
        return self.tree.predict(X)

    def _predict_proba(self, X):
        return self.tree.predict_proba(X)[:, 1]

    def _fit(self, X, y):
        self.tree.fit(X, y)


class LinearModel(BaseModel):
    def __init__(self, balanced_learning=True):
        super(LinearModel, self).__init__(normalizer='')  # use built in normalizer
        self.lr = LinearRegression(normalize=True, n_jobs=-1)
        self._threshold = 0
        self.balanced = balanced_learning

    def _predict(self, X):
        return np.asarray(self.lr.predict(X) > self._threshold, np.int8)

    def _fit(self, X, y):
        if not self.balanced:
            self.lr.fit(X, y)
        else:
            pos_num = np.sum(y)
            rate = (len(y) - pos_num) / pos_num
            weights = y * rate + 1 - y
            self.lr.fit(X, y, weights)
        self._set_threshold(y, self.lr.predict(X))

    def _predict_proba(self, X):
        return self.lr.predict(X)

    def _set_threshold(self, y_true, y_pred_prob):
        self._threshold = _get_best_threshold(y_true=y_true, y_pred_prob=y_pred_prob)


class SVM(BaseModel):
    def __init__(self, kernel='rbf', balanced_learning=True, normalizer='minmax'):
        super(SVM, self).__init__(normalizer, None)
        if balanced_learning:
            class_weight = 'balanced'
        else:
            class_weight = None
        self.svc = SVC(
            class_weight=class_weight,
            kernel=kernel,
            probability=True,
            random_state=123,
        )

    def _predict(self, X):
        return self.svc.predict(X)

    def _predict_proba(self, X):
        return self.svc.predict_proba(X)[:, 1]

    def _fit(self, X, y):
        self.svc.fit(X, y)


def _get_best_threshold(y_true, y_pred_prob):
    """

    :param y_pred_prob:
    :param y_true:
    :return: threshold.
        pred = 1, if score > threshold;
        pred = 0, otherwise;
    """
    combined = [x for x in zip(y_pred_prob, y_true)]
    combined.sort(key=lambda x: x[0])
    threshold = combined[0][0] - 0.1
    pos_len = np.sum(y_true)
    tp, fp, fn = pos_len, len(y_true) - pos_len, 0
    f1 = 2*tp / (2*tp + fp + fn)

    best_f1, best_threshold = f1, threshold
    for threshold, y in combined:
        if y == 1:
            tp -= 1
            fn += 1
        else:
            fp -= 1
        f1 = 2*tp / (2*tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    return best_threshold


__all__ = [
    'BaseEnsembleModel', 'VotingEnsemble', 'XGBoost', 'LinearEnsemble',
    'DecisionTree', 'LinearModel', 'BaseModel', 'SVM'
]

