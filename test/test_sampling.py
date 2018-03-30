from preprocess import *
from sklearn.model_selection import train_test_split
import model
import evaluation
X, y = load_data('../HTRU2/HTRU_2.csv')

all_sample_methods = [
    upsampling,
    downsampling,
    get_smote(kind='svm'),
    get_smote(kind='regular'),
    get_smote(kind='borderline1'),
    get_smote(kind='borderline2'),
]


def test_sampling_rate():
    ratio = 1
    for i, sample_method in enumerate(all_sample_methods):
        try:
            _, y_ = sample_method(X, y, ratio=ratio)
        except AssertionError:
            continue
        except ValueError as e:
            print(ratio, i)
            raise e
        assert abs(sum(y_) - ratio*(len(y_) - sum(y_))) < 2

    for ratio in (2, 4, 7):
        for i, sample_method in enumerate(all_sample_methods[:2]):
            try:
                _, y_ = sample_method(X, y, ratio=ratio)
            except AssertionError:
                continue
            assert abs(sum(y_) - ratio*(len(y_) - sum(y_))) < 10, "{}, {}".format(i, ratio)


def test_model_sampling():
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    m = model.DecisionTree()
    for sample_method in all_sample_methods:
        m.set_sample_method(sample_method)
        print(evaluation.estimate(m, X_train, X_test, y_train, y_test))
