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


