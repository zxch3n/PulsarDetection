import numpy as np
import xgboost as xgb
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from preprocess import train_test_split
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
    def __init__(self, normalizer='minmax', imbalance_method=None):
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

    def set_imbalance_method(self, method):
        self.imbalance_method = method

    def _normalize_fit(self, X):
        self.normalizer.fit(X)

    def _normalize_transform(self, X):
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
            estimators=[(i, learner) for i, learner in enumerate(learners)],
            voting='soft',
            weights=weights
        )

    def _fit(self, X, y):
        self.classifier.fit(X, y)

    def _predict(self, X):
        return self.classifier.predict(X)


class LinearEnsemble(BaseEnsembleModel):
    def __init__(self, learners, validation_rate=0.4):
        super(LinearEnsemble, self).__init__(learners)
        self.weights = np.ones(len(learners))
        self.learners = learners
        self.lr_learner = LinearRegression(normalize=True)
        self.validation_rate = validation_rate

    def _fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=123, test_set_rate=self.validation_rate)
        for learner in self.learners:
            learner.fit(X_train, y_train)
        ys = self._raw_predict(X_val)
        self.lr_learner.fit(ys, y_val)

    def _raw_predict(self, X):
        """


        :param X: the input X from outside
        :return: a concatenated matrix contains all the predictions from every sub-learner
        """
        ys = np.concatenate([ln.predict(X) for ln in self.learners], axis=1)
        return ys

    def _predict(self, X):
        return self.lr_learner.predict(self._raw_predict(X))


class XGBoost(BaseModel):
    def __init__(self, balanced_learning=True, normalizer='minmax'):
        super(XGBoost, self).__init__(normalizer)
        self.xgb = xgb.XGBClassifier(n_jobs=-1)
        self.balanced = balanced_learning

    def _predict(self, X):
        return self.xgb.predict(X)

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
            class_weight = 'balance'
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

    def _fit(self, X, y):
        self.tree.fit(X, y)


class LinearModel(BaseModel):
    def __init__(self):
        super(LinearModel, self).__init__(normalizer='')  # use built in normalizer
        self.lr = LinearRegression(normalize=True, n_jobs=-1)

    def _predict(self, X):
        return self.lr.predict(X)

    def _fit(self, X, y):
        self.lr.fit(X, y)


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
            random_state=123
        )

    def _predict(self, X):
        return self.svc.predict(X)

    def _fit(self, X, y):
        self.svc.fit(X, y)





