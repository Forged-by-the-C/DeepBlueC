import os
import glob
import pandas as pd
import numpy as np

def one_hot_ec(df: pd.DataFrame, to_oh: list):
    return pd.get_dummies(df, columns=to_oh)

def binary_ec(df, to_bin: list):
    return

def eng_features(df: pd.DataFrame):
    #Get cols to eng
    continuous = ["count_floors_pre_eq", "age","area_percentage", "height_percentage"]
    categorical = ["roof_type", "ground_floor_type", "other_floor_type", "position", "plan_configuration"]
    ordinal = []

    #Drop columns not used in model
    all_cols = categorical + ordinal + continuous
    eng_df = df[all_cols]
    #Contunious

    #Turn age of building into groups due to high screw and Nepal's changes in building codes
    age_col = eng_df["age"]
    eng_df["age_group"] = np.where(age_col < 15, 0, np.where(age_col < 35, 1, 2 ))

    #Factor hight by area of the building in order to account for "slenderness"
    hight = eng_df["count_floors_pre_eq"]

    #Categorical
    eng_df = one_hot_ec(eng_df, categorical)

    #Ordinal
    return eng_df


if __name__ == '__main__':
    pass
