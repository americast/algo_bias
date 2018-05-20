import torch
import numpy as np
import pandas as pd
import sys

BATCH_SIZE = 9

def get_yr(x):
	return x.split('/')[-1]


def process(req):
	req_new = req.loc[:, ['DateOfBirth']]
	# print(req_new)
	req_new = req_new.applymap(get_yr)
	return req_new

# Read the data
df = pd.read_csv("data/out.csv")

cols = np.array([3,7,8,9,10,13,14,15,16,17,18,22])
cols += 1
df = df[df.columns[cols]]

for i in xrange((df.shape[0])/BATCH_SIZE):
	curr = i * BATCH_SIZE
	req = df.loc[curr:curr+BATCH_SIZE, :]
	final_tensor = process(req)
	print(final_tensor["DateOfBirth"])

# print(df.shape[0])
