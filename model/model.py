import numpy as np
import random
import xgboost as xgb
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans, k_means
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import ClassifierMixin

normalizers = {'minmax': MinMaxScaler, 'standard': StandardScaler}
random.seed(100)


class BaseModel(with_metaclass(ABCMeta, ClassifierMixin, BaseEstimator)):
    @abstractmethod
    def __init__(self, normalizer_name=None, sample_method=None, sample_ratio=1.0):
        """

        :param normalizer_name:
        :param sample_method:
        """
        if normalizer_name in normalizers:
            self.normalizer = normalizers[normalizer_name]()
        else:
            self.normalizer = None
        self.sample_method = sample_method
        self.normalizer_name = normalizer_name
        self.sample_ratio = sample_ratio

    @abstractmethod
    def _predict(self, X):
        pass

    @abstractmethod
    def _predict_proba(self, X):
        pass

    @abstractmethod
    def _fit(self, X, y):
        pass

    def set_params(self, **params):
        try:
            super(BaseModel, self).set_params(**params)
        except ValueError as e:
            pass

        if self._estimator is not None:
            try:
                self._estimator.set_params(**params)
            except ValueError as e:
                pass

        return self

    @property
    def _estimator(self):
        return None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values

        if self.normalizer is not None:
            self._normalize_fit(X)
            X = self._normalize_transform(X)

        # imbalance method may use KNN/SVM to generate data
        # so it should be used after the normalization process
        if self.sample_method is not None:
            X, y = self.sample_method(X, y, ratio=self.sample_ratio)

        self._fit(X, y)
        return self

    def set_sample_ratio(self, ratio):
        self.sample_ratio = ratio

    def predict(self, X):
        X = self._normalize_transform(X)
        return self._predict(X)

    def predict_proba(self, X):
        X = self._normalize_transform(X)
        return self._predict_proba(X)

    def set_sample_method(self, method):
        self.sample_method = method
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
        super(BaseEnsembleModel, self).__init__()  # do not need normalization
        for learner in learners:
            assert isinstance(learner, BaseModel)
        self.learners = learners


class StackedEnsembleModel(BaseEnsembleModel):
    def __init__(self, learners, next_model, min_recall=0.95):
        super(StackedEnsembleModel, self).__init__(learners)
        self.thresholds = []
        self.min_recall = 0.95
        self.next_model = next_model
        self.filter_rate_ = {'predict': [], 'fit': []}
    
    def _fit(self, X, y):
        random.seed(888)
        for i, cls in enumerate(self.learners):
            X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=random.randint(1, 99999))
            cls.fit(X_train, y_train)
            pred = cls.predict_proba(X_val)
            self.thresholds[i] = self._find_threshold(y_val, pred)
            pred = cls.predict_proba(X)
            index = pred >= thresholds[i]
            X, y = X[index], y[index]
        self.next_model.fit(X, y)

    def _find_threshold(self, y_true, y_pred):
        f1_threshold = _get_best_threshold(y_true, y_pred)        
        recall_threshold = self._find_min_recall_threshold(y_true, y_pred)
        if recall_threshold > f1_threshold:
            # additional removals are not needed 
            return f1_threshold
        return recall_threshold
    
    def _find_min_recall_threshold(self, y_true, y_pred):
        combined = [x for x in zip(y_pred, y_true)]
        combined.sort(key=lambda x: x[0])
        pre = combined[0][0] - 0.1
        pos_len = np.sum(y_true)
        tp, fp, fn = pos_len, len(y_true) - pos_len, 0
        for threshold, y in combined:
            if y == 1:
                tp -= 1
                fn += 1
            else:
                fp -= 1
            recall = tp / (tp + fn)
            if recall < self.min_recall:
                return pre
            pre = threshold
        return threshold

    def _predict(self, X):
        index = np.array(True, size=len(X))
        for i, cls in enumerate(self.learners):
            pred = cls.predict_proba(_X)
            index[pred < self.thresholds[i]] = False
        y = np.array(0, size=len(X))
        y[index] = self.next_model.predict(X[index])
        self.filter_rate_['predict'].append(np.mean(index))
        return y

    def _predict_proba(self, X):
        index = np.array(True, size=len(X))
        for i, cls in enumerate(self.learners):
            pred = cls.predict_proba(_X)
            index[pred < self.thresholds[i]] = False
        y = np.array(0, size=len(X))
        y[index] = self.next_model.predict_proba(X[index])
        self.filter_rate_['predict'].append(np.mean(index))
        return y


class VotingEnsemble(BaseEnsembleModel):
    def __init__(self, learners, weights=None, random_drop_rate=0.5):
        super(VotingEnsemble, self).__init__(learners)
        self.weights = weights
        self.learners = learners
        self.random_drop_rate = random_drop_rate
        self.classifier = VotingClassifier(
            estimators=[('{}_{}'.format(learner.__class__, i), learner) for i, learner in enumerate(learners)],
            voting='soft',
            weights=weights
        )

    def _fit(self, X, y):
        X_, _, y_, _ = train_test_split(X, y, test_size=self.random_drop_rate, random_state=random.randint(1, 100000))
        self.classifier.fit(X_, y_)

    def _predict(self, X):
        return self.classifier.predict(X)

    def _predict_proba(self, X):
        return self.classifier.predict_proba(X)


class ProbaEnsemble(BaseEnsembleModel):
    def __init__(self, learners, ensemble_model, validation_rate=0.2, random_drop_rate=0.3, reuse_val=False):
        super(ProbaEnsemble, self).__init__(learners)
        self.weights = np.ones(len(learners))
        self.learners = learners
        self.lr_learner = ensemble_model
        self.random_drop_rate = random_drop_rate
        self.validation_rate = validation_rate
        self.scaler = StandardScaler()
        self.reuse_val = reuse_val

    def _fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=123, test_size=self.validation_rate)
        if self.reuse_val:
            X_train, y_train = X, y

        random.seed(123)
        for learner in self.learners:
            X_train_dropped, _, y_train_dropped, _ = train_test_split(X_train, y_train,
                                                                      random_state=random.randint(0, 100000000),
                                                                      test_size=self.random_drop_rate)
            learner.fit(X_train_dropped, y_train_dropped)
        ys = self._raw_predict(X_val)
        ys = self.scaler.fit_transform(ys)
        self.lr_learner.fit(ys, y_val)

    def _raw_predict(self, X):
        """

        :param X: the input X from outside
        :return: a concatenated matrix contains all the predictions from every sub-learner
        """
        ys = np.array([ln.predict_proba(X)[:, 1] for ln in self.learners])
        ys = ys.T
        return ys

    def _predict(self, X):
        ys = self._raw_predict(X)
        ys = self.scaler.transform(ys)
        return self.lr_learner.predict(ys)

    def _predict_proba(self, X):
        return self.lr_learner.predict_proba(self._raw_predict(X))

    @property
    def _estimator(self):
        return self.lr_learner


class LinearEnsemble(ProbaEnsemble):
    def __init__(self, learners, validation_rate=0.2, random_drop_rate=0.3, reuse_val=False):
        super(LinearEnsemble, self).__init__(learners, LinearModel(), validation_rate,
                                             random_drop_rate, reuse_val=reuse_val)


class MLPEnsemble(ProbaEnsemble):
    def __init__(self, learners, layers, activation='relu', validation_rate=0.5,
                 random_drop_rate=0.5, reuse_val=False):
        _layers = []
        for n in layers:
            if n == 0:
                break
            _layers.append(n)
        md = MLPClassifier(hidden_layer_sizes=_layers, random_state=123, activation=activation)
        super(MLPEnsemble, self).__init__(learners, md, validation_rate,
                                          random_drop_rate, reuse_val=reuse_val)

        self.layers = _layers
        self.activation = activation


class XGBoost(BaseModel):
    def __init__(self, balanced_learning=True, normalizer_name='standard', n_estimators=300,
                 scale_pos_weight=1, max_depth=4, min_child_weight=3, n_jobs=-1,
                 sample_method=None, learning_rate=0.01, nthread=-1, subsample=0.8,
                 silent=True, gamma=0.0, colsample_bytree=1, reg_alpha=0, sample_ratio=1):
        super(XGBoost, self).__init__(normalizer_name, sample_method=sample_method,
                                      sample_ratio=sample_ratio)
        self.xgb = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                     min_child_weight=min_child_weight, n_jobs=n_jobs,
                                     nthread=nthread, eval_metric='auc', seed=123,
                                     scale_pos_weight=scale_pos_weight, subsample=subsample,
                                     learning_rate=learning_rate, silent=silent, gamma=gamma,
                                     colsample_bytree=colsample_bytree, reg_alpha=reg_alpha)
        self.n_estimators = n_estimators
        self.sample_method = sample_method
        self.reg_alpha = reg_alpha
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.normalizer_name = normalizer_name
        self.max_depth = max_depth
        self.silent= silent
        self.min_child_weight = min_child_weight
        self.n_jobs = n_jobs
        self.nthread = nthread
        self.balanced_learning = balanced_learning
        self.subsample = subsample
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight

    def _predict(self, X):
        return self.xgb.predict(X)

    def _predict_proba(self, X):
        proba = self.xgb.predict_proba(X)
        return proba

    def _fit(self, X, y):
        if self.balanced_learning:
            pos_num = np.sum(y)
            neg_num = len(y) - pos_num
            self.xgb.scale_pos_weight = neg_num / pos_num
        self.xgb.fit(X, y)

    def feature_importance(self):
        return self.xgb.get_booster().get_fscore()

    @property
    def _estimator(self):
        return self.xgb


class DecisionTree(BaseModel):
    def __init__(self, balanced_learning=True, max_depth=None, sample_method=None,
                 min_samples_split=2, criterion='gini', splitter='best',
                 normalizer_name='standard', sample_ratio=1.0):
        super(DecisionTree, self).__init__(normalizer_name=normalizer_name,
                                           sample_method=sample_method,
                                           sample_ratio=sample_ratio)
        if balanced_learning:
            class_weight = 'balanced'
        else:
            class_weight = None
        self.balanced_learning = balanced_learning
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.splitter = splitter
        self.tree = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            class_weight=class_weight,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=5,
            random_state=123
        )

    def _predict(self, X):
        return self.tree.predict(X)

    def _predict_proba(self, X):
        return self.tree.predict_proba(X)

    def _fit(self, X, y):
        self.tree.fit(X, y)

    @property
    def _estimator(self):
        return self.tree


class LinearModel(BaseModel):
    def __init__(self, balanced_learning=True, sample_method=None, C=1.0, penalty='l2', tol=1e-4,
                 normalizer_name='standard', sample_ratio=1.0):
        super(LinearModel, self).__init__(normalizer_name=normalizer_name,
                                          sample_method=sample_method,
                                          sample_ratio=sample_ratio)
        if balanced_learning:
            class_weight = 'balanced'
        else:
            class_weight = None
        self.lr = LogisticRegression(penalty=penalty, C=C, n_jobs=-1, random_state=123,
                                     tol=tol, class_weight=class_weight)
        self._threshold = 0
        self.tol = tol
        self.C = C
        self.penalty = penalty
        self.balanced_learning = balanced_learning

    def _predict(self, X):
        return np.asarray(self.lr.predict(X) > self._threshold, np.int8)

    def _fit(self, X, y):
        self.lr.fit(X, y)

    def _predict_proba(self, X):
        return self.lr.predict_proba(X)

    @property
    def _estimator(self):
        return self.lr


class SVM(BaseModel):
    def __init__(self, kernel='rbf', balanced_learning=True, normalizer_name='standard',
                 sample_method=None, C=1.0, gamma='auto', sample_ratio=1.0):
        super(SVM, self).__init__(normalizer_name, sample_method=sample_method, sample_ratio=sample_ratio)
        if balanced_learning:
            class_weight = 'balanced'
        else:
            class_weight = None
        self.balanced_learning = balanced_learning
        self.normalizer_name = normalizer_name
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.svc = SVC(
            class_weight=class_weight,
            kernel=kernel,
            probability=True,
            random_state=123,
            C=C,
            gamma=gamma
        )

    def _predict(self, X):
        return self.svc.predict(X)

    def _predict_proba(self, X):
        return self.svc.predict_proba(X)

    def _fit(self, X, y):
        self.svc.fit(X, y)

    @property
    def _estimator(self):
        return self.svc


class KNN(BaseModel):
    def __init__(self, sample_method=None, n_neighbors=5,
                 normalizer_name='standard', sample_ratio=1.0):
        super(KNN, self).__init__(normalizer_name=normalizer_name, sample_method=sample_method,
                                  sample_ratio=sample_ratio)
        self.ln = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.n_neighbors = n_neighbors

    def _predict_proba(self, X):
        return self.ln.predict_proba(X)

    def _predict(self, X):
        return self.ln.predict(X)

    def _fit(self, X, y):
        self.ln.fit(X, y)

    @property
    def _estimator(self):
        return self.ln


class MultiClassesLearner(BaseModel):
    def __init__(self, binary_classifier_name, cls_params=None, sample_method=None,
                 normalizer_name='standard', balanced_learning=True, implementation='vote'):
        """

        :param binary_classifier_name:
        :param cls_params:
        :param sample_method:
        :param normalizer_name:
        :param balanced_learning:
        :param implementation: {'vote', 'linear_combined', 'cluster'}
            'vote': mean(predict_proba)
            'linear_combined': combine the result by logistic regression
            'cluster': use the cluster-chosen classifier to predict
        """
        super(MultiClassesLearner, self).__init__(normalizer_name=normalizer_name,
                                                  sample_method=sample_method)
        if cls_params is None:
            cls_params = {}
        self.models = []
        self.kmeans = None
        self.balanced_learning = balanced_learning
        self.binary_classifier_name = binary_classifier_name
        self.binary_classifier = eval(binary_classifier_name)
        self.cls_params = cls_params
        self.n_clusters = 0
        self._threshold = 0
        self.implementation = implementation

        implementations = ('vote', 'linear_combined', 'cluster')
        if implementation not in implementations:
            raise ValueError("You must choose implementation from {}".format(implementation))
        if implementation == 'linear_combined':
            raise NotImplementedError()

    def _fit(self, X, y):
        X_pos = X[y == 1]
        X_neg = X[y == 0]
        n_clusters = len(X_neg) // len(X_pos)
        self.n_clusters = n_clusters
        if n_clusters > 30:
            raise ValueError(
                "The neg to pos rate is too large, it will cause {} clusters, which is > 30".format(n_clusters)
            )
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=123, n_jobs=-1)
        self.kmeans.fit(X_neg)
        Cls = self.binary_classifier
        for i in range(n_clusters):
            X_i_neg = X_neg[self.kmeans.labels_ == i]
            cls = Cls(**self.cls_params)
            X_i, y_i = np.concatenate([X_pos, X_i_neg]), np.array([1]*len(X_pos) + [0]*len(X_i_neg))
            X_i, y_i = shuffle(X_i, y_i)
            cls.fit(X_i, y_i)
            self.models.append(cls)

        scores = self._predict_proba(X)
        self._threshold = _get_best_threshold(y, scores[:, 1])

    def _predict_proba(self, X):
        if len(self.models) == 0:
            raise ValueError("Must fit before predict")
        if self.kmeans is None:
            raise ValueError("Must fit before predict")

        if self.implementation == 'vote':
            y = 0
            for model in self.models:
                y += model.predict_proba(X)
            y /= len(self.models)
            return y

        elif self.implementation == 'cluster':
            cluster_indexes = self.kmeans.predict(X)
            y = np.zeros(len(X))
            for i in range(self.n_clusters):
                model_input = X[cluster_indexes == i]
                if len(model_input) == 0:
                    continue
                pred = self.models[i].predict_proba(model_input)
                y[cluster_indexes == i] = pred
            return np.array([1 - y, y]).T

    def _predict(self, X):
        if len(self.models) == 0:
            raise ValueError("Must fit before predict")
        if self.kmeans is None:
            raise ValueError("Must fit before predict")

        if self.implementation == 'vote':
            scores = self._predict_proba(X)
            return np.array((scores[:, 1] > self._threshold), dtype=np.int)

        elif self.implementation == 'cluster':
            cluster_indexes = self.kmeans.predict(X)
            y = np.zeros(len(X))
            for i in range(self.n_clusters):
                model_input = X[cluster_indexes == i]
                if len(model_input) == 0:
                    continue
                pred = self.models[i].predict(model_input)
                y[cluster_indexes == i] = pred
            return y


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
    'DecisionTree', 'LinearModel', 'BaseModel', 'SVM', 'MultiClassesLearner',
    'KNN', 'ProbaEnsemble', 'MLPEnsemble'
]

