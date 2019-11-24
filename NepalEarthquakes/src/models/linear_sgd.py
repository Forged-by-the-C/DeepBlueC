import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
# for combining the preprocess with model training
from sklearn.pipeline import make_pipeline
# for optimizing the hyperparameters of the pipeline
from sklearn.model_selection import RandomizedSearchCV

from src.utils.model_wrapper import model_wrapper

class sgd(model_wrapper):

    def train(self, X,y):
        '''
        input X: numpy.ndarray of shape (n_smaples, n_features)
        input y: numpy.ndarray of shape (n_samples, )
        output: trained model
        '''
        pipe = make_pipeline(SGDClassifier(random_state=2018))
        param_grid = {'sgdclassifier__penalty': ['none', 'l2', 'l1', 'elasticnet']}
        clf = RandomizedSearchCV(pipe, param_grid, scoring='f1_micro', n_iter=4, cv=5, verbose=1, n_jobs=-1)
        clf.fit(X, y)
        print("Best Params: {}".format(clf.best_params_))
        return clf

if __name__ == "__main__":
    mod = sgd({"model":"sgd"})
    mod.train_and_score()
    #mod.load_and_score()
    #mod.load_and_predict_submission()
