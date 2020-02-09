import pandas as pd
import numpy as np
#from src.features.feat_eng import rf_features

def split_df(whole_df: pd.DataFrame, ratio: list, interim_loc: str):
    # Set random seed for repeatability
    np.random.seed(2016)

    # Sample from whole df then split on provided ratios
    df_len = len(whole_df)
    first_chunk_end = int(df_len * ratio[0])
    third_chunk_start = int(df_len * (1.0 - ratio[2]))
    train, validate, test = np.split(whole_df.sample(frac=1), [first_chunk_end, third_chunk_start])

    #Validate same number of rows as well as all buildings accoutned for
    assert df_len == sum([len(train), len(validate), len(test)])
    assert set(whole_df.index) == set(train.index) | set(validate.index) | set(test.index)

    #save split data into csvs
    train.to_csv(interim_loc + "train.csv")
    validate.to_csv(interim_loc + "val.csv")
    test.to_csv(interim_loc + "test.csv")
    return

def combine_vals_labels(label_df: pd.DataFrame, value_df: pd.DataFrame, primary_key: str):
    #Set index to primary_key
    label_df.set_index(primary_key, inplace=True)
    value_df.set_index(primary_key, inplace=True)
    
    #return merge on primary key
    return label_df.merge(value_df, left_index=True, right_index=True)

def gen_primary_key(df, cols_to_combine, primary_key_name="pk"):
   df[primary_key_name] = df[cols_to_combine[0]].astype(str)
   for col in cols_to_combine[1:]:
        df[primary_key_name] = df[primary_key_name].str.cat(df[col].astype(str), sep="_")

def feat_eng(df):
    out_df = df[["pk","ndvi_ne","ndvi_nw","ndvi_se","ndvi_sw"]].copy()
    out_df.fillna(0, inplace=True)
    return out_df

if __name__ == '__main__':
    # Fixed vars for data locations and [train, val, test] splits for data
    split_ratio = [.8, .1, .1]
    train_vals_loc = "../../Data/raw/dengue_features_train.csv"
    train_labels_loc = "../../Data/raw/dengue_labels_train.csv"
    submit_data_loc = "../../Data/raw/dengue_features_test.csv"
    interim_loc = "../../Data/interim/"

    PRIMARY_KEY = "pk" 
    #Combine the values with their labels
    #not needed if data
    train_vals_df = pd.read_csv(train_labels_loc)
    train_features_df = pd.read_csv(train_vals_loc)
    submit_features_df = pd.read_csv(submit_data_loc)
    df_list = [train_features_df, train_vals_df, submit_features_df]
    for df in df_list:
        gen_primary_key(df, cols_to_combine=["city", "year", "weekofyear"], primary_key_name=PRIMARY_KEY)

    whole_df = combine_vals_labels(feat_eng(train_features_df), 
            train_vals_df[[PRIMARY_KEY,"total_cases"]], PRIMARY_KEY)

    submit_features_df = feat_eng(submit_features_df)
    submit_features_df.set_index(PRIMARY_KEY, inplace=True)
    submit_features_df.to_csv("../../Data/interim/submit_vals.csv")
    #Pass combined df to split and save as csvs in processed file
    split_df(whole_df, split_ratio, interim_loc)
