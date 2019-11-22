import os
import pandas as pd
import pickle
import src.utils.data_helper as data_w
import src.utils.dir_helper as dir_w
from sklearn.ensemble import RandomForestClassifier
import src.features.rf_feat_eng as rf_features
from sklearn.metrics import f1_score

class model_wrapper():

    def __init__(self, file_name_list = ["rf", "12", "100"]):
        file_name = ""
        for name in file_name_list:
            file_name += name + "_"
        self.model_file_path = dir_w.construct_dir_path(project_dir="NepalEarthquakes",
                sub_dir="model") + file_name + ".pkl"

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
        with open(self.model_file_path, 'wb+') as f:
            pickle.dump(self.clf, f)

    def load_model(self):
        with open(self.model_file_path, 'rb+') as f:
            self.clf = pickle.load(f)

    def load_and_predict_submission(self):
        self.load_model(self.model_file_path)
        f_df = self.grab_submission_data()
        X = f_df.values
        g = self.clf.predict(X)
        y_df = pd.DataFrame(g, index=f_df.index, columns=["damage_grade"])
        y_df.to_csv('submission.csv')

    def load_and_score(self):
        self.load_model(model_file_path)
        X,y = self.grab_data(split="val")
        g = self.clf.predict(X)
        print(f1_score(y_true=y, y_pred=g, average='micro'))

    def train_and_score(self):
        X,y = self.load_data("train")
        self.clf = self.train(X,y)
        g = self.clf.predict(X)
        print("Training Score: {}".format(f1_score(y_true=y, y_pred=g, average='micro')))
        X,y = self.load_data("val")
        g = self.clf.predict(X)
        print("Val Score: {}".format(f1_score(y_true=y, y_pred=g, average='micro')))
        self.save_model()

if __name__=='__main__':
    pass
