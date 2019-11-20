import os
import pandas as pd

def construct_path(sub_dir, data_set):
    ''' 
    Data/<sub_dir>/<data_set>.csv
    input sub_dir: str
    input data_set: str
    output: full path to data_set.csv
    '''
    cwd_path = os.getcwd().split('/')
    path_to_file = []
    for d in cwd_path:
        path_to_file.append(d)
        if d == 'NepalEarthquakes':
            break
    path_to_file.append('Data')
    path_to_file.append(sub_dir)
    path_to_file.append(data_set + '.csv')
    file_path = os.path.join(*path_to_file)
    return '/' + file_path

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
    return load_df(construct_path(sub_dir, data_set))

if __name__=='__main__':
    pass
