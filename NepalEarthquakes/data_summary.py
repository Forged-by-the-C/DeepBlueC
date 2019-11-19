import os
import pandas as pd

def get_dir_csvs(path):
    dir_list = os.listdir(path)
    csv_list = []
    for f in dir_list:
        if f.split('.')[-1] == "csv":
            csv_list.append(path+f)  
    return csv_list

def print_csv_summary(csv_path):
    df = pd.read_csv(csv_path)
    print("-"*20)
    print("Filename: {}".format(csv_path))
    print("CSV Shape: {}".format(df.shape))
    print("Columns: {}".format(list(df.columns)))
    print("="*10)
    print(df.head())
    print("-"*20)

def explore_data():
    data_dir = "Data/"
    csv_list = get_dir_csvs(data_dir)
    for csv in csv_list:
        print_csv_summary(csv)

if __name__=='__main__':
    explore_data()
