import os
import pandas as pd
import pickle
import src.utils.data_helper as data_w
import src.utils.dir_helper as dir_w
from sklearn.ensemble import RandomForestClassifier
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
        features_df = data_w.grab_data("raw", "test_values")
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
        print("Cross Val Score on {} is {:.4f}".format(self.param_string, f1_score(y_true=y, y_pred=g, average='micro')))
        print(self.gen_conf_matrix(y, g))

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

    def train_and_score(self, n_iter=1, cv=5, n_jobs=1, save_model=True):
        '''
        input n_iter: int, number of training iterations if doing a hyper parameter search
        input cv: int, number of cross folds to trian on
        input n_jobs: int, number of processoers to use if doing a hyper parameter seacrch
                            -1 indicates using all processors
        input save_model: boolean
        return time_to_train: float, seconds took to train
        rerturn val_score: float, score on cross validation
        '''
        X,y = self.load_data("train")
        tic = time.time()
        self.clf = self.train(X,y, n_iter, cv, n_jobs)
        if save_model:
            self.save_model()
        time_to_train = time.time() - tic
        print("{} time to train {} iters: {:.0f} seconds".format(
            self.param_dict[list(self.param_dict.keys())[0]],
            n_iter, time_to_train))
        g = self.clf.predict(X)
        print("Training Score: {}".format(f1_score(y_true=y, y_pred=g, average='micro')))
        X,y = self.load_data("val")
        g = self.clf.predict(X)
        val_score = f1_score(y_true=y, y_pred=g, average='micro')
        print("Val Score: {}".format(f1_score(y_true=y, y_pred=g, average='micro')))
        if 0:
        #if "model" in self.param_dict.keys():
            self.print_cv_results()
        return time_to_train, val_score 

if __name__=='__main__':
    pass
