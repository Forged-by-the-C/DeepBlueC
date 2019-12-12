import os
import glob
import pandas as pd
import numpy as np
import category_encoders as ce
import pickle

def one_hot_ec(df: pd.DataFrame, to_oh: list):
    return pd.get_dummies(df, columns=to_oh)

def binary_ec(df, to_bin: list):
    return

def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def categorical_trim(df: pd.DataFrame, submit_df:pd.DataFrame, dependent: str, cat_cols: list, ratio: float):
    replaced = {}
    for col in cat_cols:
        group_vals = df.groupby(col)[dependent].sum()
        group_perc = group_vals / group_vals.sum()
        len_all_values = len(group_perc)
        avg_pct = 1.0 / len_all_values
        drop_pct = avg_pct * ratio
        print("Drop from: " + group_perc.index.name + " if less than : " + str(drop_pct))
        items_to_group = []
        for item in group_perc.index:
            if group_perc.loc[item] < (avg_pct * ratio):
                items_to_group.append(str(item))
        len_to_replace = len(items_to_group)
        if (len_to_replace > 1):
            replaced[col] = "|".join(items_to_group)
            print("Replacing " + str(len_to_replace) + " of " + str(len_all_values))
            to_repl_reg = "|".join(items_to_group)

            #Replace in both dataframes
            df[col] = df[col].astype(str).str.replace(to_repl_reg, "other")
            submit_df[col] = submit_df[col].astype(str).str.replace(to_repl_reg, "other")
        else:
            print("No items to change")

    save_model(replaced, "../features/trim_dict.pkl")
    return df, submit_df

def rf_features(eng_df: pd.DataFrame, submit_df:pd.DataFrame, dependent_col: str, to_skip: list, num_cats: list):
    #Drop skip list
    eng_df = eng_df.drop(to_skip, axis=1)
    submit_df = submit_df.drop(to_skip, axis=1)

    #Get list of categorical columns, only based on type
    categorical = list(eng_df.select_dtypes("object").columns)
    categorical = categorical + num_cats

    #Categorical trim from dict
    eng_df, submit_df = categorical_trim(eng_df, submit_df, dependent_col, categorical, .2)


    #Check for harighly cardinal and use binary encoding
    to_binary = []
    for var in categorical:
        if eng_df[var].nunique() > 10:
            to_binary.append(var)
            print("Use Binary Encoder on ", var)

    ##Binary

    ##Drop dependent
    dep_col_ser = eng_df[dependent_col].copy()
    eng_df = eng_df.drop(dependent_col, axis=1)

    # instantiate an encoder - here we use Binary()
    ce_binary = ce.BinaryEncoder(cols=to_binary)

    ce_binary.fit(eng_df)

    #Save binary encoder
    save_model(ce_binary, "../features/ce_model1.pkl")

    #Use binary encoder
    eng_df = ce_binary.transform(eng_df)
    submit_df = ce_binary.transform(submit_df)

    #Add dependent column back
    eng_df[dependent_col] = dep_col_ser

    #One Hot
    one_hot_cols = set(categorical) - set(to_binary)
    eng_df = pd.get_dummies(eng_df, columns=one_hot_cols)
    submit_df = pd.get_dummies(submit_df, columns=one_hot_cols)


    #Turn age of building into groups due to high screw and Nepal's changes in building codes
    age_col_eng = eng_df["age"]
    age_col_sub = submit_df["age"]
    eng_df = eng_df.assign(age_groups = np.where(age_col_eng < 15, 0, np.where(age_col_eng < 35, 1, 2 )))
    submit_df = submit_df.assign(age_groups=np.where(age_col_sub < 15, 0, np.where(age_col_sub < 35, 1, 2)))

    #Factor hight by area of the building in order to account for "slenderness"
    height = eng_df["count_floors_pre_eq"]
    area = eng_df["area_percentage"]
    eng_df = eng_df.assign(slenderness=(height / area))

    height_sub = submit_df["count_floors_pre_eq"]
    area_sub = submit_df["area_percentage"]
    submit_df = submit_df.assign(slenderness=(height_sub / area_sub))

    return eng_df, submit_df


if __name__ == '__main__':
    #Test
    #test_df = pd.read_csv("../../Data/raw/test_values.csv", index_col="building_id")
    #rev_one_hot(test_df)
    #eng_df = rf_features(test_df, "damage_grade", ["geo_level_2_id", "geo_level_3_id"], ["geo_level_1_id"])
    pass
