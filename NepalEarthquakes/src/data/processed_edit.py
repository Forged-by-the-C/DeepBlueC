import glob
import pandas as pd


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
    print("Collapsed ", len(one_hot_cols), "collumns.")
    to_return.drop(list(one_hot_cols), inplace=True, axis=1)

    # Drop list col
    rev_one_hot_df.drop("rev_oh_list", inplace=True, axis=1)

    # Merge in oh str snd count
    to_return = to_return.merge(rev_one_hot_df, left_index=True, right_index=True)

    return to_return

def edit_to_processed(interim_df :pd.DataFrame):
    interim_df.drop("has_secondary_use", axis=1, inplace=True)
    interim_df = rev_one_hot(interim_df, "has_secondary_use")
    interim_df = rev_one_hot(interim_df, "has_superstructure")
    return interim_df

if __name__=='__main__':

    for item in glob.glob('../../Data/interim/*.csv'):
        processed_df = edit_to_processed(pd.read_csv(item))
        processed_df.to_csv(item.replace("interim", "processed"), index=False)
