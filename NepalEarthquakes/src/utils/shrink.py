def reduce_mem_usage(df_to_shrink):
    #Stolen from Kaggle...
    start_mem_usg = df_to_shrink.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in df_to_shrink.columns:
        if df_to_shrink[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", df_to_shrink[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = df_to_shrink[col].max()
            mn = df_to_shrink[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df_to_shrink[col]).all():
                NAlist.append(col)
                df_to_shrink[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = df_to_shrink[col].fillna(0).astype(np.int64)
            result = (df_to_shrink[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df_to_shrink[col] = df_to_shrink[col].astype(np.uint8)
                    elif mx < 65535:
                        df_to_shrink[col] = df_to_shrink[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df_to_shrink[col] = df_to_shrink[col].astype(np.uint32)
                    else:
                        df_to_shrink[col] = df_to_shrink[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df_to_shrink[col] = df_to_shrink[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df_to_shrink[col] = df_to_shrink[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df_to_shrink[col] = df_to_shrink[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df_to_shrink[col] = df_to_shrink[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                df_to_shrink[col] = df_to_shrink[col].astype(np.float32)

            # Print new column type
            print("dtype after: ", df_to_shrink[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df_to_shrink.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return df_to_shrink, NAlist


if __name__ == '__main__':
    pass