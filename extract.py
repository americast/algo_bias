import pandas as pd
import numpy as np

df = pd.read_csv("data/raw.csv")
cols = df.columns.values

req_cols =[cols[i] for i in (3,7,8,9,10,13,14,15,16,17,18,22)]

for col in req_cols:
	unique_rows = df[col].unique()
	i = 1
	for each_row in unique_rows:
		df[col] = df[col] .replace([each_row], i)
		i+=1
		
print(df)
df.to_csv("out.csv")

#Numeralise Agency_Text

# print(req_cols)