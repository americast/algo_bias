import torch
import numpy as np
import pandas as pd
import sys

BATCH_SIZE = 9

def get_db_yr(x):
	return int(x.split('/')[-1])

def get_sc_yr(x):
	return int(x.split('/')[-1].split(' ')[0]) + 100


def process_sc(req):
	req_new = req.loc[:,['Screening_Date']]
	req_new = req_new.applymap(get_sc_yr)
	return req_new

def process_dob(req):
	req_new = req.loc[:, ['DateOfBirth']]
	req_new = req_new.applymap(get_db_yr)
	return req_new

def get_age(req):
	req["Age"]=req.loc[:,['Screening_Date']].sub(req['DateOfBirth'], axis=0)["Screening_Date"]
	# print req["Age"]
	return req
	# pass


# Read the data
df = pd.read_csv("data/out.csv")

cols = np.array([3,7,8,9,10,13,14,15,16,17,18,22])
cols += 1
df = df[df.columns[cols]]

for i in xrange((df.shape[0])/BATCH_SIZE):
	curr = i * BATCH_SIZE
	req = df.loc[curr:curr+BATCH_SIZE, :]
	req["DateOfBirth"] = process_dob(req)["DateOfBirth"]
	req["Screening_Date"] = process_sc(req)["Screening_Date"]
	# print(req["Screening_Date"])
	req = get_age(req)
	print(req)
	# print(final_tensor[:,["DateOfBirth","Screening_Date"]])

# print(df.shape[0])
