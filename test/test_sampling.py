from preprocess import *


def test_sampling():
    X, y = load_data()
    from sklearn.model_selection import train_test_split
    import model
    import evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    m = model.LinearModel().set_sample_method(upsampling)
    print(evaluation.estimate(m, X_train, X_test, y_train, y_test))

    m = model.LinearModel().set_sample_method(downsampling)
    print(evaluation.estimate(m, X_train, X_test, y_train, y_test))

    m = model.LinearModel().set_sample_method(get_smote(kind='svm'))
    print(evaluation.estimate(m, X_train, X_test, y_train, y_test))

    m = model.LinearModel().set_sample_method(get_smote(kind='regular'))
    print(evaluation.estimate(m, X_train, X_test, y_train, y_test))

    m = model.LinearModel().set_sample_method(get_smote(kind='borderline1'))
    print(evaluation.estimate(m, X_train, X_test, y_train, y_test))
