#To be run from the docker container
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.utils.model_wrapper import model_wrapper
import autosklearn.classification
import time

class auto_sk(model_wrapper):

        def train(self, X,y):
            '''
            input X: numpy.ndarray of shape (n_smaples, n_features)
            input y: numpy.ndarray of shape (n_samples, )
            output: trained model
            '''
            tic = time.time()
            print("Starting Training")
            input()
            clf = autosklearn.classification.AutoSklearnClassifier()
            clf.fit(X, y)
            print("Training finished. Took {:.0f}s".format(time.time() - tic))
            return clf


if __name__ == "__main__":
    mod = auto_sk({"auto":"sk"})
    mod.train_and_score()
    #mod.load_and_score()
    #mod.load_and_predict_submission()
