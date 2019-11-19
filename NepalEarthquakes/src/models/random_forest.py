from sklearn.ensemble import RandomForestClassifier

training_data_values = "Data/train_values.csv"
training_data_labels = "Data/train_labels.csv"


def grab_train_data():
    '''
    output X: numpy.ndarray of shape (n_smaples, n_features)
    output y: numpy.ndarray of shape (n_samples, )
    '''
    pass

def grab_t_data():
    pass

def example():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=4,
                                        n_informative=2, n_redundant=0,
                                        random_state=0, shuffle=False)
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                          random_state=0)
    clf.fit(X, y)

    print(clf.feature_importances_)
    print(clf.predict([[0, 0, 0, 0]]))

if __name__=='__main__':
    example()
