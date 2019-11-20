import pandas as pd

def one_hot_ec(df: pd.DataFrame, to_oh: list):
    return pd.get_dummies(df, columns=to_oh)

def binary_ec(df, to_bin: list):
    return

def eng_features(df: pd.DataFrame, categorical: list, ordinal: list, continuous: list, drop_eng: list):
    #Drop columns not used in model
    df = df[df.columns.difference(drop_eng)]

    #Contunious


    #Categorical
    df = one_hot_ec(df, categorical)

    print(df)
    #Ordinal

    return df


if __name__ == '__main__':
    # Set data locaitons
    train_loc = "../../Data/interim/train.csv"
    processed_loc = "../../Data/processed/"
    continuous = ["count_floors_pre_eq", "age","area_percentage", "height_percentage"]
    categorical = ["geo_level_1_id", "roof_type", "ground_floor_type", "other_floor_type", "position", "plan_configuration"]
    ordinal = []
    drop_eng = ['geo_level_2_id', 'geo_level_3_id']
    train_df = pd.read_csv(train_loc)
    eng_features(train_df, categorical, ordinal, continuous, drop_eng)
