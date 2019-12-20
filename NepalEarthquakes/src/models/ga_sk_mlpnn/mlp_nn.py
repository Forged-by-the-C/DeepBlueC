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
        clf = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=10)
        clf.fit(X, y)
        return clf

    def run_ga(self, load_population=False):
        pass

    def train_and_score(self, n_iter=1, cv=2, n_jobs=-1, save_model=True):
        X,y = self.load_data("train")
        tic = time.time()
        self.clf = self.train(X,y, n_iter, cv, n_jobs)
        if save_model:
            self.save_model()
        g = self.clf.predict(X)
        ## Log the results
        self.results_dict["time_to_train"] = time.time() - tic
        self.results_dict["n_iter"] = n_iter
        self.results_dict["cross_folds"] = cv
        self.results_dict["n_jobs"] = n_jobs
        self.results_dict["training_score"] = f1_score(y_true=y, y_pred=g, average='micro') 
        X,y = self.load_data("val")
        g = self.clf.predict(X)
        self.results_dict["val_score"] = f1_score(y_true=y, y_pred=g, average='micro')
        if hasattr(self.clf, 'cv_results_'):
            cvres = self.clf.cv_results_
            self.results_dict["cv_results"] = list(zip(cvres["mean_test_score"], cvres["params"]))
        self.log_results()

if __name__ == "__main__":
    mod = mlp({"single":"mlp"})
    mod.train_and_score(save_model=True)
    #mod.load_and_score()
    #mod.load_and_predict_submission()
