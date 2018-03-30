from preprocess import *
from sklearn.model_selection import train_test_split
import evaluation
import model
X, y = load_data('../HTRU2/HTRU_2.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)


def test_multi_classes():
    m = model.MultiClassesLearner(model.DecisionTree, {'max_depth': 5})
    print(evaluation.estimate(m, X_train, X_test, y_train, y_test))

