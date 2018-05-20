import torch
import numpy as np
import pandas as pd

# Read the data
df = pd.read_csv("data/out.csv")

cols = np.array([3,7,8,9,10,13,14,15,16,17,18,22])
cols += 1
df = df[df.columns[cols]]

print(df)
