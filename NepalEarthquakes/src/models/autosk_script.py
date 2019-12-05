#To be run from the docker container
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.utils.model_wrapper import model_wrapper
import autosklearn.classification

run_time_s = 1 * 5 * 60

class auto_sk(model_wrapper):

        def train(self, X,y):
            '''
            input X: numpy.ndarray of shape (n_smaples, n_features)
            input y: numpy.ndarray of shape (n_samples, )
            output: trained model
            '''
            print("Starting Training")
            clf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=run_time_s)
            clf.fit(X, y)
            return clf


if __name__ == "__main__":
    mod = auto_sk({"auto":"sk1"})
    #mod.train_and_score()
    mod.load_and_score()
    #mod.load_and_predict_submission()
