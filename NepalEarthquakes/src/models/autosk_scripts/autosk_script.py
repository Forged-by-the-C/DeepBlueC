#To be run from the docker container
import numpy as np
import pandas as pd
from src.utils.model_wrapper import model_wrapper
import autosklearn.classification
import autosklearn.metrics

run_time_s = 1 * 15 * 60

class auto_sk(model_wrapper):

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
        print("Starting Training")
        clf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=run_time_s)
        clf.fit(X, y, metric=autosklearn.metrics.f1_micro)
        return clf


if __name__ == "__main__":
    mod = auto_sk({"auto":"sk1"})
    mod.train_and_score()
    #mod.load_and_score()
    #mod.load_and_predict_submission()
