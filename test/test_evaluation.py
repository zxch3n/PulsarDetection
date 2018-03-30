import model
import evaluation
import preprocess
from sklearn.model_selection import train_test_split

X, y = preprocess.load_data('./HTRU2/HTRU_2.csv')
X, y = X[:100], y[:100]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)


def test_cross_validation():
    for md in [
        model.SVM(),
        model.KNN(),
        model.XGBoost(),
        model.LinearModel(),
        model.DecisionTree()
    ]:
        try:
            evaluation.cross_validation(md, X, y, scoring='all', n_jobs=1, n_splits=2)
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
