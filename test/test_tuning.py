import model
import numpy as np
import evaluation
import preprocess
from sklearn.model_selection import train_test_split

X, y = preprocess.load_data('./HTRU2/HTRU_2.csv')
X, y = X[:100], np.array([1]*20 + [0]*80)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)


def test_best_param_search():
    md = model.SVM()
    best_params, result = evaluation.best_param_search(
        md,
        X=X,
        y=y,
        params=[
            {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']},
            {'gamma': [0.005, 0.0125, 0.02, 0.04, 0.08, 0.1]}
        ],
        n_jobs=1
    )
    for i in range(len(result) - 1):
        assert result['test_score'].diff().abs().sum() > 0.0001
    print(best_params)
    print(result)


def test_tuning():
    models = {
        'DecisionTree': model.DecisionTree,
        'SVM': model.SVM,
        'LR': model.LinearModel,
        'XGBoost': model.XGBoost
    }

    models_params = {
        'SVM': [
            {'kernel': ['rbf', 'poly'], 'C': [0.001, 0.01, 0.1, 1, 4, 16, 32]},
            {'gamma': [0.0075, 0.0125, 0.25]},
            {'normalizer_name': ['standard', 'minmax']},
            {'sample_method': [None, preprocess.downsampling, preprocess.upsampling],
             'sample_ratio': [1 / 3, 1 / 4, 1 / 5]},
            {'balanced_learning': [True, False]}
        ],
        'DecisionTree': [
            {'max_depth': [None, 3, 5, 7, 9], 'sample_method': [None, preprocess.downsampling, preprocess.upsampling],
             'sample_ratio': [1 / 4]},
            {'min_samples_split': [2, 4, 6, 8, 10, 12]},
            {'balanced_learning': [True, False]},
            {'sample_ratio': [1, 1 / 3, 1 / 4, 1 / 5]},
            {'normalizer_name': ['standard', 'minmax']},
        ],
        'XGBoost': [
            {'n_estimators': [50, 80, 100, 120, 180], 'learning_rate': [0.05, 0.1, 0.2, 0.3, 0.4]},
            {'max_depth': [100, 3, 5, 8, 11], 'min_child_weight': [1, 2, 4, 8, 16]},
            {'gamma': [0.0, 0.01, 0.05, 0.1, 0.2, 0.4]},
            {'n_estimators': [400, 800, 1200, 1600], 'learning_rate': [0.03, 0.05, 0.08, 0.12, 0.25]},
            {'balanced_learning': [True, False]},
            {'sample_method': [None, preprocess.downsampling, preprocess.upsampling],
             'sample_ratio': [1 / 3, 1 / 4, 1 / 5]},
            {'normalizer_name': ['standard', 'minmax']},
        ],
        'LR': [
            {'C': [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0], 'penalty': ['l2', 'l1']},
            {'balanced_learning': [True, False]},
            {'sample_method': [None, preprocess.downsampling, preprocess.upsampling],
             'sample_ratio': [1 / 3, 1 / 4, 1 / 5]},
            {'tol': [1e-4, 1e-3, 1e-5]},
            {'normalizer_name': ['standard', 'minmax']},
        ]
    }

    dict_best_params = {}
    dict_best_estimators = {}
    dict_scores = {}

    for model_name, md in models.items():
        print(model_name)
        param = models_params[model_name]
        best_param, df_score, best_estimator_ = evaluation.best_param_search(
            estimator=md(),
            params=param,
            X=X,
            y=y
        )
        dict_best_params[model_name] = best_param
        dict_best_estimators[model_name] = best_estimator_
        dict_scores[model_name] = df_score
        print(best_param)
        print()




