import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier
# for combining the preprocess with model training
from sklearn.pipeline import make_pipeline
# for optimizing the hyperparameters of the pipeline
from sklearn.model_selection import RandomizedSearchCV

from src.utils.model_wrapper import model_wrapper

'''
http://drivendata.co/blog/richters-predictor-benchmark/
'''

train_space = 15
cross_folds = 3

class extra_trees(model_wrapper):

    def train(self, X,y, n_iter, cv, n_jobs):
        '''
        input X: numpy.ndarray of shape (n_smaples, n_features)
        input y: numpy.ndarray of shape (n_samples, )
        input n_iter: int, number of training iterations if doing a hyper parameter search
        input cv: int, number of cross folds to trian on
        input n_jobs: int, number of processoers to use if doing a hyper parameter seacrch
                            -1 indicates using all processors
        output: trained model
        '''
        pipe = make_pipeline(ExtraTreesClassifier(random_state=2018))
        param_grid = {'extratreesclassifier__n_estimators':range(50,150),
                        'extratreesclassifier__min_samples_leaf':range(3,10)}
        clf = RandomizedSearchCV(pipe, param_grid, scoring='f1_micro', n_iter=n_iter,
                cv=cv, verbose=1, n_jobs=n_jobs)
        clf.fit(X, y)
        self.results_dict["best_params"] = clf.best_params_
        return clf

if __name__ == "__main__":
    mod = extra_trees({"model":"et"})
    #mod.train_and_score(n_iter=train_space, cv=cross_folds, n_jobs=-1, save_model=True)
    #mod.log_results()
    #mod.load_and_score()
    mod.load_and_predict_submission()
