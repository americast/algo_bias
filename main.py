from __future__ import print_function
import torch
import argparse
import numpy as np
import pandas as pd
import sys

BATCH_SIZE = 8192
ACCESS_PARAM = "Scale_ID"
EPOCHS = 10
LEARNING_RATE = 0.01
# SGD_MOMENTUM = 0.5

count = 0

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = torch.nn.Linear(8, 4)
        self.fc2 = torch.nn.Linear(4, 1)

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        # x = F.dropout(x, training=self.training)
        # loss = torch.nn.MSELoss()
        # output = loss(out_tensor, result)
        # return F.log_softmax(x, dim=1)
        return x

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
	# # print( req)
	# for i in xrange(BATCH_SIZE):
	# 	if not req.at[count,ACCESS_PARAM] == 7:
	# 		to_drop.append(i)
	# count+=1
	# # print(to_drop)
	# req = req.drop(req.index[to_drop])
	array_now = req.loc[:,["Agency_Text","Sex_Code_Text","Ethnic_Code_Text","Language","LegalStatus","CustodyStatus","MaritalStatus","Age"]].values
	array_now = torch.cuda.FloatTensor(array_now)
	array_now.cuda()
	# print(array_now)
	# print("rockstud")

	result_now = req.loc[:,["RawScore"]].values
	result_now =torch.cuda.FloatTensor(result_now)
	result_now.cuda()
	return array_now, result_now


# Read the data
df = pd.read_csv("data/out.csv")

cols = np.array([3,7,8,9,10,13,14,15,16,17,18,20,22])
cols += 1
df = df[df.columns[cols]]

parser = argparse.ArgumentParser(description='Algo_bias')
parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, metavar='N',
                        help='input batch size for training (default: 9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=EPOCHS, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=LEARNING_RATE, metavar='LR',
                    help='learning rate (default: 0.01)')
# parser.add_argument('--momentum', type=float, default=SGD_MOMENTUM, metavar='M',
#                     help='SGD momentum (default: 0.5)')


args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)

print("Would you like to restore a previously saved model? (y/n)")
choice = raw_input()

if (choice=='y' or choice=='Y'):
	path = raw_input("Enter path: ")
	the_model.load_state_dict(torch.load(path))

print("\n")

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

for j in xrange(EPOCHS):
	losses=0.0
	global count
	count = 0
	for i in xrange((df.shape[0])/BATCH_SIZE):
		curr = i * BATCH_SIZE
		req = df.loc[curr:curr+BATCH_SIZE-1, :]
		req["DateOfBirth"] = process_dob(req)["DateOfBirth"]
		req["Screening_Date"] = process_sc(req)["Screening_Date"]
		# print(req["Screening_Date"])
		req = get_age(req)
		# print(req)
		final_tensor, result = create_tensor(req)
		# print(final_tensor,"\n", result)

		# hidden_tensor = torch.nn.Linear(8,4)(final_tensor)
		# out_tensor = torch.nn.Linear(4,1)(hidden_tensor)
		output = model(final_tensor)
		loss = torch.nn.MSELoss()
		loss_here = loss(output, result)
		loss_here.backward()
		optimizer.step()
		# print("Loss: ", loss_here.data[0])
		losses+=loss_here.data[0]
		# print(final_tensor[:,["DateOfBirth","Screening_Date"]])
		count+=BATCH_SIZE
		print("Batches done: "+str(count)+"/"+str(df.shape[0]),end="\r")
	losses/=count
	print("\nIteration "+str(j+1)+"/"+str(EPOCHS)+" done")
	print("\nLoss: "+ str(losses))
	torch.save(model.state_dict(), "checkpoints/"+str(j))

	# print(df.shape[0])
