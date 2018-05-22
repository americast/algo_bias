import torch
import numpy as np
import pandas as pd
import sys

BATCH_SIZE = 9
ACCESS_PARAM = "Scale_ID"

count = 0

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

def create_tensor(req):
	to_drop=[]
	global count
	# print( req)
	for i in xrange(BATCH_SIZE):
		if not req.at[count,ACCESS_PARAM] == 7:
			to_drop.append(i)
		count+=1
	print(to_drop)
	req = req.drop(req.index[to_drop])
	array_now = req.loc[:,["Agency_Text","Sex_Code_Text","Ethnic_Code_Text","Language","LegalStatus","CustodyStatus","MaritalStatus","Age"]].values
	array_now = torch.FloatTensor(array_now)
	# print(array_now)
	# print("rockstud")

	result_now = req.loc[:,["RawScore"]].values
	result_now =torch.FloatTensor(result_now)
	return array_now, result_now


# Read the data
df = pd.read_csv("data/out.csv")

cols = np.array([3,7,8,9,10,13,14,15,16,17,18,20,22])
cols += 1
df = df[df.columns[cols]]

for i in xrange((df.shape[0])/BATCH_SIZE):
	curr = i * BATCH_SIZE
	req = df.loc[curr:curr+BATCH_SIZE-1, :]
	req["DateOfBirth"] = process_dob(req)["DateOfBirth"]
	req["Screening_Date"] = process_sc(req)["Screening_Date"]
	# print(req["Screening_Date"])
	req = get_age(req)
	# print(req)
	final_tensor, result = create_tensor(req)
	print(final_tensor,"\n", result)

	hidden_tensor = torch.nn.Linear(8,4)(final_tensor)
	out_tensor = torch.nn.Linear(4,1)(hidden_tensor)
	loss = torch.nn.MSELoss()
	output = loss(out_tensor, result)
	output.backward()
	# print(final_tensor[:,["DateOfBirth","Screening_Date"]])

# print(df.shape[0])
