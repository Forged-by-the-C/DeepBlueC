import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
# for combining the preprocess with model training
from sklearn.pipeline import make_pipeline
# for optimizing the hyperparameters of the pipeline
from sklearn.model_selection import RandomizedSearchCV

from src.utils.model_wrapper import model_wrapper

'''
http://drivendata.co/blog/richters-predictor-benchmark/
'''

class mlp(model_wrapper):

    def train(self, X,y):
        '''
        input X: numpy.ndarray of shape (n_smaples, n_features)
        input y: numpy.ndarray of shape (n_samples, )
        output: trained model
        '''
        pipe = make_pipeline(MLPClassifier(random_state=2018))
        param_grid = {'mlpclassifier__solver':['lbfgs', 'sgd', 'adam']}
        clf = RandomizedSearchCV(pipe, param_grid, scoring='f1_micro', n_iter=3, cv=2, verbose=1, n_jobs=-1)
        clf.fit(X, y)
        print("Best Params: {}".format(clf.best_params_))
        return clf

if __name__ == "__main__":
    mod = mlp({"model":"mlp"})
    mod.train_and_score()
    #mod.load_and_score()
    #mod.load_and_predict_submission()
