import os
import pandas as pd
import pickle
import src.utils.data_helper as data_w
import src.utils.dir_helper as dir_w
from sklearn.ensemble import RandomForestClassifier
import src.features.rf_feat_eng as rf_features
from sklearn.metrics import f1_score
import time

class model_wrapper():

    def __init__(self, param_dict = {"name":"init"}):
        self.param_dict = param_dict
        self.param_string = self.gen_param_string(self.param_dict)
        self.gen_model_file_path()

    def gen_param_string(self, param_dict):
        param_string = ""
        for k in param_dict:
            param_string += "{}_{}_".format(k, param_dict[k])
        return param_string

    def gen_model_file_path(self):
        self.model_file_path = dir_w.construct_dir_path(project_dir="NepalEarthquakes",
                sub_dir="models") + self.param_string + ".pkl"

    def load_data(self, split="train"):
        '''
        input split: str, of ["train", "val", "test"]
        output X: numpy.ndarray of shape (n_smaples, n_features)
        output y: numpy.ndarray of shape (n_samples, )
        '''
        features_df, label_series = data_w.grab_data("interim", split)
        features_df = rf_features.eng_features(features_df)
        X = features_df.values
        y = label_series.values
        return X, y

    def trim(self, X, y,num=100):
        return X[:num], y[:num]

    def grab_submission_data(self):
        features_df = data_w.grab_data("raw", "test_values")
        features_df = rf_features.eng_features(features_df)
        return features_df

    def train(self, X,y):
        '''
        input X: numpy.ndarray of shape (n_smaples, n_features)
        input y: numpy.ndarray of shape (n_samples, )
        output: trained model
        '''
        n_estimators = 100
        max_depth = 3
        clf = RandomForestClassifier(n_estimators=n_estimators, 
                max_depth=max_depth, random_state=1)
        clf.fit(X, y)
        return clf

    def save_model(self):
        with open(self.model_file_path, 'wb') as f:
            pickle.dump(self.clf, f)

    def load_model(self):
        with open(self.model_file_path, 'rb+') as f:
            self.clf = pickle.load(f)

    def load_and_predict_submission(self):
        self.load_model()
        f_df = self.grab_submission_data()
        X = f_df.values
        g = self.clf.predict(X)
        y_df = pd.DataFrame(g, index=f_df.index, columns=["damage_grade"])
        y_df.to_csv('submission.csv')

    def load_and_score(self):
        self.load_model()
        X,y = self.load_data(split="val")
        g = self.clf.predict(X)
        print(f1_score(y_true=y, y_pred=g, average='micro'))

    def print_cv_results(self):
        cvres = self.clf.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(mean_score, params)

    def train_and_score(self):
        X,y = self.load_data("train")
        tic = time.time()
        self.clf = self.train(X,y)
        print("Time to train: {:.0f} seconds".format(time.time() - tic))
        g = self.clf.predict(X)
        print("Training Score: {}".format(f1_score(y_true=y, y_pred=g, average='micro')))
        X,y = self.load_data("val")
        g = self.clf.predict(X)
        print("Val Score: {}".format(f1_score(y_true=y, y_pred=g, average='micro')))
        self.save_model()
        if "model" in self.param_dict.keys():
            self.print_cv_results()

if __name__=='__main__':
    pass
