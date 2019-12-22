import json
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import time

import src.utils.data_helper as data_w
import src.utils.dir_helper as dir_w

class model_wrapper():

    def __init__(self, param_dict = {"name":"init"}):
        self.param_dict = param_dict
        self.results_dict = {}
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

    def gen_conf_plot_file_path(self):
        return dir_w.construct_dir_path(project_dir="NepalEarthquakes",
                sub_dir="models") + "conf_plot_" + self.param_string + ".png"

    def load_data(self, split="train"):
        '''
        input split: str, of ["train", "val", "test"]
        output X: numpy.ndarray of shape (n_smaples, n_features)
        output y: numpy.ndarray of shape (n_samples, )
        '''
        features_df, label_series = data_w.grab_data("interim", split)
        #features_df = rf_features(features_df.merge(label_series,left_index=True,
        #                            right_index=True), "damage_grade", 
        #                            to_skip=["geo_level_2_id", "geo_level_3_id"], 
        #                            numm_cats=["geo_level_1_id"]) 
        X = features_df.values
        y = label_series.values
        return X, y

    def trim(self, X, y,num=100):
        return X[:num], y[:num]

    def grab_submission_data(self):
        features_df = data_w.grab_data("interim", "submit_vals")
        #features_df = rf_features.eng_features(features_df)
        return features_df

    def train(self, X,y, n_iter=1, cv=5, n_jobs=1):
        '''
        input X: numpy.ndarray of shape (n_smaples, n_features)
        input y: numpy.ndarray of shape (n_samples, )
        input n_iter: int, number of training iterations if doing a hyper parameter search
        input cv: int, number of cross folds to trian on
        input n_jobs: int, number of processoers to use if doing a hyper parameter seacrch
                            -1 indicates using all processors
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
        self.results_dict["val_score"] = f1_score(y_true=y, y_pred=g, average='micro')
        conf_matrix = self.gen_conf_matrix(y, g)
        self.results_dict["confusion_matrix"] = conf_matrix.tolist()
        self.log_results()
        self.plot_conf_matrix(conf_matrix)

    def gen_conf_matrix(self, y_true, y_pred):
        '''
        input y_true: numpy ndarray
        input y_pred: numpy ndarray
        return: numpy ndarray, conf matrix 
        '''
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred, normalize='all')

    def plot_conf_matrix(self, conf_matrix):
        '''
        input: numpy ndarray, conf_matrix
        '''
        plt_path = self.gen_conf_plot_file_path()
        import matplotlib.pyplot as plt
        #from sklearn.metrics import plot_confusion_matrix
        #disp = plot_confusion_matrix(self.clf, X, y,normalize='all')
        from sklearn.metrics import ConfusionMatrixDisplay
        display_labels = [i + 1 for i in range(0, conf_matrix.shape[0])]
        ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=display_labels).plot()
        plt.savefig(plt_path)

    def print_cv_results(self):
        cvres = self.clf.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(mean_score, params)

    def log_results(self, print_to_screen=True):
        json_filename = dir_w.construct_dir_path(project_dir="NepalEarthquakes",
                sub_dir="models") + "results.json"
        if os.path.exists(json_filename):
            with open(json_filename, 'r') as outfile:
                    write_dict = json.load(outfile)
        else: 
            write_dict = {}
        if self.param_string in write_dict:
            write_dict[self.param_string].update(self.results_dict)
        else:
            write_dict[self.param_string] = self.results_dict
        if print_to_screen:
            for k in write_dict[self.param_string]:
                print("{} : {}".format(k, write_dict[self.param_string][k]))
        with open(json_filename, 'w') as outfile:
                json.dump(write_dict, outfile, indent=6)

    def train_and_score(self, n_iter=1, cv=2, n_jobs=-1, save_model=True):
        '''
        input n_iter: int, number of training iterations if doing a hyper parameter search
        input cv: int, number of cross folds to trian on
        input n_jobs: int, number of processoers to use if doing a hyper parameter seacrch
                            -1 indicates using all processors
        input save_model: boolean
        '''
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

if __name__=='__main__':
    pass
