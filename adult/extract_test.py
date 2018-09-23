import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("adult1.csv")
df1=pd.read_csv("adult.csv")
cols = df.columns.values

req_cols_sparse = [cols[i] for i in (1,3,5,6,7,8,9)]

for col in req_cols_sparse:
    unique_rows = df[col].unique()
    i = 1
    for each_row in unique_rows:
        df[col + str(i)] = (df[col] == each_row) * 1
        i+=1

col = "native-country"
unique_rows = df1[col].unique()
i = 1
for each_row in unique_rows:
    if each_row=="Holand-Netherlands":
        df[col + str(i)] = 0
    else:   
        df[col + str(i)] = (df[col] == each_row) * 1
    i+=1


df2 = (df["income"] == ' <=50K.') * 1
df = df.drop(columns = ["income"])
df = df.drop(columns = ["native-country"])
df["income"] = df2

df = df.drop(columns = req_cols_sparse)

df.to_csv("adult_test.csv", index=False)
print(df.shape)