import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/people.csv")

cols = df.columns.values

req_cols =[cols[i] for i in (4,5,7,8,9,10,11,16,24,25)]

for col in req_cols:
	unique_rows = df[col].unique()
	i = 1
	for each_row in unique_rows:
		df[col + str(i)] = (df[col] == each_row) * 1
		i+=1

# col = 'c_charge_desc'
# i = 1
# for each_row in df[col].unique():
# 	df[col] = df[col] .replace([each_row], i)
# 	i+=1

lower_threshold = 4.0
middle_threshold = 7.0

df1 = (df["decile_score"] > lower_threshold) * 1
# df2 = (df["decile_score"] > middle_threshold) * 1 

# df_add = df1.add(df2, fill_value=0)

df["decile_score"] = df1

cols = np.delete(cols,[13])

df = df.drop(columns = cols)

train, test = train_test_split(df, test_size=(1.0/6.0))

df.to_csv("data/out.csv", index=False)
train.to_csv("data/out-train.csv", index=False)
test.to_csv("data/out-test.csv",  index=False) 

print(train.shape)
print(test.shape)
print(df.shape)