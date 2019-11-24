import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
# for combining the preprocess with model training
from sklearn.pipeline import make_pipeline
# for optimizing the hyperparameters of the pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# for preprocessing the data
from sklearn.preprocessing import StandardScaler

from src.utils.model_wrapper import model_wrapper

'''
http://drivendata.co/blog/richters-predictor-benchmark/
'''

class random_forest(model_wrapper):

    def train(self, X,y):
        '''
        input X: numpy.ndarray of shape (n_smaples, n_features)
        input y: numpy.ndarray of shape (n_samples, )
        output: trained model
        '''
        pipe = make_pipeline(RandomForestClassifier(random_state=2018))
        param_grid = {'randomforestclassifier__n_estimators': range(90,150),
                              'randomforestclassifier__min_samples_leaf':range(1,10)}
        #clf = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
        clf = RandomizedSearchCV(pipe, param_grid, scoring='f1_micro', n_iter=20, cv=5, n_jobs=-1)
        clf.fit(X, y)
        print("Best Params: {}".format(clf.best_params_))
        return clf

if __name__ == "__main__":
    mod = random_forest({"model":"rf"})
    mod.train_and_score()
    #mod.load_and_score()
    #mod.load_and_predict_submission()
