import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier

from src.utils.model_wrapper import model_wrapper
import src.utils.data_helper as data_w

'''
http://drivendata.co/blog/richters-predictor-benchmark/
'''

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
        clf = ExtraTreesClassifier(random_state=2018, 
                n_estimators=320, max_features="sqrt",
                                  min_samples_leaf=3, n_jobs=-1)
        clf.fit(X, y)
        return clf

if __name__ == "__main__":
    mod = extra_trees({"single":"et"})
    #mod.train_and_score(save_model=True)
    #mod.load_and_score()
    mod.load_and_predict_submission()
