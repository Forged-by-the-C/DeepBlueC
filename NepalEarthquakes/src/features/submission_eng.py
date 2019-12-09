import pandas as pd
import category_encoders as ce
import pickle
import numpy as np

def load_model(file_path):
    with open(file_path, 'rb+') as f:
        model = pickle.load(f)
    return model

def eng_submit(eng_df: pd.DataFrame, dependent_col: str, to_skip: list, num_cats: list):
    # Drop skip list
    eng_df = eng_df.drop(to_skip, axis=1)


    # Get list of categorical columns, only based on type
    categorical = list(eng_df.select_dtypes("object").columns)
    categorical = categorical + num_cats

    # Categorical will be from cols
    #eng_df = categorical_trim(eng_df, dependent_col, categorical, .2)

    # Check for harighly cardinal and use binary encoding
    to_binary = []
    for var in categorical:
        if eng_df[var].nunique() > 10:
            to_binary.append(var)
            print("Use Binary Encoder on ", var)

    ##Binary
    # instantiate an encoder - here we use Binary()
    ce_binary = load_model("ce_model.pkl")

    # Use binary encoder
    eng_df = ce_binary.transform(eng_df)

    # One Hot
    one_hot_cols = set(categorical) - set(to_binary)
    eng_df = pd.get_dummies(eng_df, columns=one_hot_cols)

    # Turn age of building into groups due to high screw and Nepal's changes in building codes
    age_col = eng_df["age"]
    eng_df = eng_df.assign(age_groups=np.where(age_col < 15, 0, np.where(age_col < 35, 1, 2)))

    # Factor hight by area of the building in order to account for "slenderness"
    height = eng_df["count_floors_pre_eq"]
    area = eng_df["area_percentage"]
    eng_df = eng_df.assign(slenderness=(height / area))
    # Ordinal

    return eng_df


    return submit_df

if __name__ == '__main__':
    trans_df = pd.read_csv("../../Data/raw/test_values.csv", index_col="building_id")
    sub_df = eng_submit(trans_df, "damage_grade", ["geo_level_2_id", "geo_level_3_id"], ["geo_level_1_id"])
    sub_df.to_csv("../../Data/interim/submit_vals.csv")
    pass