import os
import pandas as pd
import pickle
import src.utils.grab_data as gd
from sklearn.ensemble import RandomForestClassifier
import src.features.rf_feat_eng as rf_features
from sklearn.metrics import f1_score

def grab_data(split="train", processed=False):
    '''
    input split: str, of ["train", "val", "test"]
    output X: numpy.ndarray of shape (n_smaples, n_features)
    output y: numpy.ndarray of shape (n_samples, )
    '''
    features_df, label_series = gd.grab_data("interim", "train")
    features_df = rf_features(features_df)
    X = features_df.values
    y = label_series.values
    return X, y

def grab_submission_data():
    features_df = gd.grab_data("raw", "test_values")
    features_df = features_df[training_cols]
    return features_df

def train_rf(X,y):
    '''
    input X: numpy.ndarray of shape (n_smaples, n_features)
    input y: numpy.ndarray of shape (n_samples, )
    output: trained model
    '''
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                          random_state=0)
    clf.fit(X, y)
    return clf

def save_model(clf, file_path):
    '''
    Pickels model

    input clf: trained model
    '''
    with open(file_path, 'wb+') as f:
        pickle.dump(clf, f)

def train_and_save():
    X,y = grab_data("train")
    clf = train_rf(X,y)
    save_model(clf, 'scratch.pkl')

def load_model(file_path):
    with open(file_path, 'rb+') as f:
        clf = pickle.load(f)
    return clf

def load_and_predict_submission(model_file_path):
    clf = load_model(model_file_path)
    f_df = grab_submission_data()
    X = f_df.values
    g = clf.predict(X)
    y_df = pd.DataFrame(g, index=f_df.index, columns=["damage_grade"])
    y_df.to_csv('submission.csv')

def load_and_score(model_file_path):
    clf = load_model(model_file_path)
    X,y = grab_data(split="val")
    g = clf.predict(X)
    print(f1_score(y_true=y, y_pred=g, average='micro'))

def train():
    X,y = grab_data("train", processed=True)
    clf = train_rf(X,y)
    g = clf.predict(X)
    print("Training Score: {}".format(f1_score(y_true=y, y_pred=g, average='micro')))
    X,y = grab_data("val", processed=True)
    g = clf.predict(X)
    print("Val Score: {}".format(f1_score(y_true=y, y_pred=g, average='micro')))
    save_model(clf, 'first_feature_eng.pkl')

if __name__=='__main__':
    #load_and_predict_submission('scratch.pkl')
    #model_file_path = 'scratch.pkl'
    train()
