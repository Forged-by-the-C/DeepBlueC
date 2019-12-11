import pandas as pd

df = pd.read_csv('submission.csv')
print("Num Samples: {}".format(df.shape[0]))
print(df['damage_grade'].value_counts())
print(df['damage_grade'].value_counts()/float(df.shape[0]))
