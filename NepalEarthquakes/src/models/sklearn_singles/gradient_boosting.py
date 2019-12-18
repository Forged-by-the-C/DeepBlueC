import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
# for combining the preprocess with model training
from sklearn.pipeline import make_pipeline
# for optimizing the hyperparameters of the pipeline
from sklearn.model_selection import RandomizedSearchCV

from src.utils.model_wrapper import model_wrapper

'''
http://drivendata.co/blog/richters-predictor-benchmark/
'''

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
        clf = GradientBoostingClassifier(
                    max_depth=20,
                    max_features="sqrt",
                    min_samples_leaf=15,
                    min_samples_split=19,
                    n_estimators= 22
                )
        clf.fit(X, y)
        return clf

if __name__ == "__main__":
    mod = gradient_boosting({"single":"gb"})
    mod.train_and_score(n_jobs=-1, save_model=True)
    #mod.load_and_score()
    #mod.load_and_predict_submission()
