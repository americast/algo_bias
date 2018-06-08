import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
	return req

df = pd.read_csv("data/raw.csv")
# print(df["RawScore"].max())
# df = df[df.DisplayText != 'Risk of Failure to Appear']
# df = df[df.DisplayText != 'Risk of Recidivism']
# df = df[df.DisplayText != 'Risk of Violence']
cols = df.columns.values

df["DateOfBirth"] = process_dob(df)["DateOfBirth"]
df["Screening_Date"] = process_sc(df)["Screening_Date"]
df = get_age(df)

req_cols =[cols[i] for i in (3,7,8,13,14,15,16)]

for col in req_cols:
	unique_rows = df[col].unique()
	i = 1
	for each_row in unique_rows:
		df[col + str(i)] = (df[col] == each_row) *1
		i+=1

lower_threshold = 4.0
middle_threshold = 7.0

df1 = (df["DecileScore"] > lower_threshold) * 1
df2 = (df["DecileScore"] > middle_threshold) * 1 

df_add = df1.add(df2, fill_value=0)

df["DecileScore"] = df_add

df = df.drop(columns=["Person_ID", "AssessmentID", "Case_ID", "Agency_Text", "LastName", "FirstName", "MiddleName", "Sex_Code_Text", "Ethnic_Code_Text", "DateOfBirth", "ScaleSet_ID", "ScaleSet", "AssessmentReason", "Language", "LegalStatus", "CustodyStatus", "MaritalStatus", "Screening_Date", "RecSupervisionLevel", "RecSupervisionLevelText", "Scale_ID", "DisplayText", "RawScore", "ScoreText", "AssessmentType", "IsCompleted", "IsDeleted"])

train, test = train_test_split(df, test_size=(1.0/6.0))

df.to_csv("data/out.csv")
train.to_csv("data/out-train.csv")
test.to_csv("data/out-test.csv") 

print(train.shape)
print(test.shape)
print(df.shape)