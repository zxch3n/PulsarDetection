from preprocess import *
from sklearn.model_selection import train_test_split
import evaluation
import model
X, y = load_data('./HTRU2/HTRU_2.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)


def test_multi_classes():
    m = model.MultiClassesLearner(model.DecisionTree, {'max_depth': 5})
    print(evaluation.estimate(m, X_train, X_test, y_train, y_test))


def test_xgboost():
    m = model.XGBoost(n_jobs=1)
    print(evaluation.cross_validation(m, X, y, n_jobs=1))


def test_logistic_regression():
    m = model.LinearModel()
    X = np.random.normal(size=(1000, 10))
    y = np.array(X.sum(axis=1) > 0, dtype=np.int8)
    scores = evaluation.cross_validation(m, X, y, n_jobs=1)
    assert np.all(scores > 0.9)


