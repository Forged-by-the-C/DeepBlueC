import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier
# for combining the preprocess with model training
from sklearn.pipeline import make_pipeline
# for optimizing the hyperparameters of the pipeline
from sklearn.model_selection import RandomizedSearchCV

from src.utils.model_wrapper import model_wrapper
import src.utils.data_helper as data_w

'''
http://drivendata.co/blog/richters-predictor-benchmark/
'''

importance_threshold = 0.95
data_sub_dir = "interim"

class extra_trees(model_wrapper):

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
        clf = ExtraTreesClassifier(random_state=2018, n_estimators=50, n_jobs=-1)
        clf.fit(X, y)
        return clf

    def prune_features(self):
        self.train_and_score(save_model=True)
        #self.load_model()
        features_df, label_series = data_w.grab_data(data_sub_dir, "val")
        df = pd.DataFrame(list(zip(features_df.columns, self.clf.feature_importances_)), 
                columns=["feature", "importance"])
        df.sort_values("importance", ascending=False, inplace=True)
        df["cum_import"] = df["importance"].cumsum()
        df.to_csv('feature_imp.csv', index=False)
        important_cols = df[df.cum_import < importance_threshold]["feature"].values
        orig_count = len(features_df.columns) 
        new_count = len(important_cols)
        pruned_count = orig_count - new_count
        print("Original Feature Count: {} New Count: {} Number Pruned: {}".format(orig_count, new_count, pruned_count))
        val_path = data_w.construct_data_path(data_sub_dir, "val")
        features_df[important_cols].merge(label_series, left_index=True, right_index=True).to_csv(val_path)
        features_df, label_series = data_w.grab_data(data_sub_dir, "train")
        train_path = data_w.construct_data_path(data_sub_dir, "train")
        features_df[important_cols].merge(label_series, left_index=True, right_index=True).to_csv(train_path)

if __name__ == "__main__":
    mod = extra_trees({"feat":"et"})
    #mod.train_and_score(n_iter=train_space, cv=cross_folds, n_jobs=-1, save_model=True)
    mod.prune_features()
    #mod.load_and_score()
    #mod.load_and_predict_submission()
