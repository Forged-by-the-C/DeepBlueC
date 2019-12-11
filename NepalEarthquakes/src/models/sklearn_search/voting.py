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
        self.results_dict["model_list"] = []
        for f in os.listdir(model_dir):
            if 'model' in f:
                with open((model_dir+f), 'rb+') as c:
                    _clf = pickle.load(c)
                model_name = f.split('.')[0]
                self.results_dict["model_list"].append(model_name)
                self.clf_tuple_list.append((model_name,_clf.best_estimator_)) 

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
        self.gen_clf_tuple_list()
        clf = VotingClassifier(estimators=self.clf_tuple_list, voting='hard', n_jobs=n_jobs)
        clf = clf.fit(X, y)
        return clf

if __name__ == "__main__":
    mod = voting({"ensemble":"voting"})
    #mod.gen_clf_tuple_list()
    mod.train_and_score(n_jobs=-1, save_model=True)
    #mod.load_and_score()
    #mod.load_and_predict_submission()
