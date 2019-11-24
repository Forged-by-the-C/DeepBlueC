import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import f1_score
import src.utils.dir_helper as dir_w
from sklearn.ensemble import VotingClassifier
from src.utils.model_wrapper import model_wrapper

'''
http://drivendata.co/blog/richters-predictor-benchmark/
'''

class voting(model_wrapper):

    def gen_clf_tuple_list(self):
        model_dir = dir_w.construct_dir_path(project_dir="NepalEarthquakes",
                sub_dir="models")
        self.clf_tuple_list = []
        for f in os.listdir(model_dir):
            if 'model' in f:
                with open((model_dir+f), 'rb+') as c:
                    _clf = pickle.load(c)
                self.clf_tuple_list.append((f.split('.')[0],_clf.best_estimator_)) 

    def train(self, X,y):
        '''
        input X: numpy.ndarray of shape (n_smaples, n_features)
        input y: numpy.ndarray of shape (n_samples, )
        output: trained model
        '''
        self.gen_clf_tuple_list()
        clf = VotingClassifier(estimators=self.clf_tuple_list, voting='hard')
        clf = clf.fit(X, y)
        return clf

if __name__ == "__main__":
    mod = voting({"ensemble":"voting"})
    #mod.gen_clf_tuple_list()
    #mod.train_and_score()
    #mod.load_and_score()
    mod.load_and_predict_submission()
