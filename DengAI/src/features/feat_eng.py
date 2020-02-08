import os
import glob
import pandas as pd
import numpy as np
import category_encoders as ce
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

def one_hot_ec(df: pd.DataFrame, to_oh: list):
    return pd.get_dummies(df, columns=to_oh)

def rf_features(df):

    return df

def binary_ec(df, to_bin: list):
    return

def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def prune(eng_df, submit_df):
    print("*"*5, "Pruning Model: ","*"*5,)
    print("Features before Prune: ", len(submit_df.columns))
    X = eng_df.drop("damage_grade", axis=1)
    y = eng_df["damage_grade"]
    model = ExtraTreesClassifier(n_estimators=50)
    selector = model.fit(X, y)
    feat_model = SelectFromModel(selector, prefit=True, threshold=.0075)
    new_eng = X.loc[:,feat_model.get_support()]
    new_eng = new_eng.assign(damage_grade = eng_df["damage_grade"])
    new_submit_df = submit_df.loc[:,feat_model.get_support()]
    print("Features after Prune: ", len(new_submit_df.columns))
    return new_eng, new_submit_df

def rev_one_hot(df, prefix):
    # Copy df for return
    to_return = df.copy()

    # Get columns from df that use o-h prefix
    one_hot_cols = df.columns[df.columns.str.contains(prefix)]
    if len(one_hot_cols) == 0:
        print("None found to collapse.")
        return

    # Make new df with only those columns
    one_hot_df = to_return[one_hot_cols]

    # Turn 0/1 into booleans
    boolean_df = one_hot_df.apply(lambda x: x > 0)

    # Use that to get list of all values shown
    # This is only a list for those that have more than one value which is atypical for one hot
    rev_one_hot_df = pd.DataFrame(boolean_df.apply(lambda x: list(one_hot_cols[x.values]), axis=1),
                                  columns=["rev_oh_list"])

    # Get count as a feature
    kwargs_cnt = {str(prefix + "_count"): rev_one_hot_df["rev_oh_list"].str.len()}
    rev_one_hot_df = rev_one_hot_df.assign(**kwargs_cnt)

    # Turn into a string for hashing
    kwargs_str = {prefix: rev_one_hot_df["rev_oh_list"].apply(", ".join)}
    rev_one_hot_df = rev_one_hot_df.assign(**kwargs_str)

    # Merge back into df

    # Drop the cols that were collapsed
    print("Collapsed ", len(one_hot_cols), "columns.")
    #to_return.drop(list(one_hot_cols), inplace=True, axis=1)

    # Drop list col
    rev_one_hot_df.drop("rev_oh_list", inplace=True, axis=1)

    # Merge in oh str snd count
    to_return = to_return.merge(rev_one_hot_df, left_index=True, right_index=True)

    return to_return

def edit_to_processed(df :pd.DataFrame):
    interim_df = df.copy()
    interim_df.drop("has_secondary_use", axis=1, inplace=True)
    interim_df = rev_one_hot(interim_df, "has_secondary_use")
    interim_df = rev_one_hot(interim_df, "has_superstructure")
    return interim_df

def categorical_trim(start_df: pd.DataFrame, submit_df:pd.DataFrame, dependent: str, cat_cols: list, ratio: float):
    df = start_df.copy()
    replaced = {}
    #Reverse one hot
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

    #save_model(replaced, "../features/trim_dict.pkl")
    return df, submit_df

def old_rf_features(eng_df: pd.DataFrame, submit_df:pd.DataFrame, dependent_col: str, to_skip: list, num_cats: list, james:list):
    #Drop skip list
    eng_df = eng_df.drop(to_skip, axis=1)
    submit_df = submit_df.drop(to_skip, axis=1)
    eng_df = edit_to_processed(eng_df)
    submit_df = edit_to_processed(submit_df)


    #Get list of categorical columns, only based on type
    categorical = list(eng_df.select_dtypes("object").columns)
    categorical = categorical + num_cats

    #Remove those used for james enc
    categorical = list(set(categorical) - set(james))

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
    #save_model(ce_binary, "../features/ce_model1.pkl")

    #Use binary encoder
    eng_df = ce_binary.transform(eng_df)
    submit_df = ce_binary.transform(submit_df)

    for var in james:
        print("Use James Encoder on ", var)

    #Use james encoder
    ce_james = ce.JamesSteinEncoder(cols=james)
    ce_james.fit(eng_df, dep_col_ser)

    # Use binary encoder
    eng_df = ce_james.transform(eng_df)
    submit_df = ce_james.transform(submit_df)

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

    #Prune to the new features
    eng_df, submit_df = prune(eng_df, submit_df)

    return eng_df, submit_df


if __name__ == '__main__':
    #Test
    #test_df = pd.read_csv("../../Data/raw/test_values.csv", index_col="building_id")
    #rev_one_hot(test_df)
    #eng_df = rf_features(test_df, "damage_grade", ["geo_level_2_id", "geo_level_3_id"], ["geo_level_1_id"])
    pass
