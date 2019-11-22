import os
import pandas as pd
from src.utils.dir_helper import construct_dir_path

def construct_data_path(sub_dir, data_set):
    ''' 
    Data/<sub_dir>/<data_set>.csv
    input sub_dir: str
    input data_set: str
    output: full path to data_set.csv
    '''
    return construct_dir_path(project_dir="NepalEarthquakes",
            sub_dir="Data") + "{}/{}.csv".format(sub_dir, data_set)

def load_df(file_path):
    '''
    input file_path: str, full file path, such as output from contruct_path()
    output features_df: pandas dataframe, index building_id, columns of features
    output label_series: pandas series, index building_id, entries of damage_grade
    '''
    df = pd.read_csv(file_path, index_col=0)
    label_col = "damage_grade"
    features = list(df.columns)
    if label_col in features:
        features.remove(label_col)
        features_df = df[features].copy() 
        label_series = df[label_col].copy()
        return features_df, label_series
    else: 
        return df

def grab_data(sub_dir, data_set):
    return load_df(construct_data_path(sub_dir, data_set))

if __name__=='__main__':
    pass
