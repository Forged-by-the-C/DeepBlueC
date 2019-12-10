'''
Attempting to implement

https://scikit-optimize.github.io/notebooks/sklearn-gridsearchcv-replacement.html
'''

import numpy as np
from skopt import gp_minimize
import time

def f(x):
    return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) *
            np.random.randn() * 0.1)

def simple():
    res = gp_minimize(f, [(-2.0, 2.0)])
    print(res.fun)

from skopt import BayesSearchCV
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def minimal_ex():
    X, y = load_digits(10, True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

    # log-uniform: understand as search over p = exp(x) by varying x
    tic = time.time()
    opt = BayesSearchCV(
        SVC(),
        {
            'C': (1e-6, 1e+6, 'log-uniform'),  
            'gamma': (1e-6, 1e+1, 'log-uniform'),
            'degree': (1, 8),  # integer valued parameter
            'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
        },
        n_iter=32,
        n_jobs=6,
        scoring='f1_micro',
        cv=3
        )

    opt.fit(X_train, y_train)

    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(X_test, y_test))
    print("best params: {}".format(opt.best_params_))
    print("time to evaluate: {:.0f}".format(time.time()-tic))

if __name__=='__main__':
    #simple()
    minimal_ex()
