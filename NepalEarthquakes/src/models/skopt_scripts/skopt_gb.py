import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
# for combining the preprocess with model training
from sklearn.pipeline import make_pipeline
# for optimizing the hyperparameters of the pipeline
from skopt import BayesSearchCV

from src.utils.model_wrapper import model_wrapper

'''
http://drivendata.co/blog/richters-predictor-benchmark/
'''

train_space = 15
cross_folds = 3
n_jobs=6

class gradient_boosting(model_wrapper):

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
        clf = BayesSearchCV(
            GradientBoostingClassifier(),
            {
                'n_estimators': (10, 400),  
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 20),
                'max_depth': (1, 20),
                'max_features': ['auto', 'sqrt', 'log2'],  # categorical parameter
            },
            n_iter=n_iter,
            scoring='f1_micro',
            n_jobs=n_jobs,
            cv=cv
            )
        self.results_dict["total_iterations"] = clf.total_iterations
        clf.fit(X, y)
        self.results_dict["best_params"] = clf.best_params_
        return clf

if __name__ == "__main__":
    mod = gradient_boosting({"sko":"gb"})
    mod.train_and_score(n_iter=train_space, cv=cross_folds, n_jobs=n_jobs, save_model=True)
    #mod.load_and_score()
    #mod.load_and_predict_submission()
