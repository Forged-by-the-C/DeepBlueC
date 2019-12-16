import pandas as pd
import numpy as np
from src.features.feat_eng import rf_features


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


if __name__ == '__main__':
    # Fixed vars for data locations and [train, val, test] splits for data
    split_ratio = [.8, .1, .1]
    train_vals_loc = "../../Data/raw/train_values.csv"
    train_labels_loc = "../../Data/raw/train_labels.csv"
    submit_data_loc = "../../Data/raw/test_values.csv"
    interim_loc = "../../Data/interim/"

    #Combine the values with their labels
    #not needed if data
    whole_df = combine_vals_labels(pd.read_csv(train_vals_loc), pd.read_csv(train_labels_loc), "building_id")

    submit_df = pd.read_csv(submit_data_loc, index_col="building_id")

    whole_df, submit_df = rf_features(whole_df, submit_df, "damage_grade",
                                to_skip=[],
                                num_cats=["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"],
                                james = ["land_surface_condition", "has_secondary_use", "has_superstructure"])


    submit_df.to_csv("../../Data/interim/submit_vals.csv")
    #Pass combined df to split and save as csvs in processed file
    split_df(whole_df, split_ratio, interim_loc)
