import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/people.csv")
# print(df)
cols = df.columns.values

req_cols =[cols[i] for i in (4,5,7, 8,9,10,11,16,24)]

print("req_cols: ", req_cols)
df = df.dropna(axis='rows', subset = req_cols)

for col in req_cols:
    # print(col)
    unique_rows = df[col].unique()
    i = 1
    for each_row in unique_rows:
        df[col + str(i)] = (df[col] == each_row) * 1
        i+=1

col = cols[25]
i = 1
for each_row in df[col].unique():
    df[col] = df[col] .replace([each_row], i)
    i+=1


lower_threshold = 4.0
middle_threshold = 7.0

df1 = (df["decile_score"] > lower_threshold) * 1
df2 = (df["decile_score"] > middle_threshold) * 1 

df_add = df1.add(df2, fill_value=0)

df["decile_score"] = df_add

# cols_to_keep = req_cols + [13, 25]


cols = np.delete(cols,[13, 25])
df = df.drop(columns = cols)

# df =  df.dropna(axis='rows')
train, test = train_test_split(df, test_size=(1.0/6.0))

df.to_csv("data/out.csv", index=False)
train.to_csv("data/out-train.csv", index=False)
test.to_csv("data/out-test.csv",  index=False) 

print(train.shape)
print(test.shape)
print(df.shape)