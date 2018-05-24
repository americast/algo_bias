from __future__ import print_function
import torch
import argparse
import numpy as np
import pandas as pd
import sys
import os

BATCH_SIZE = 8192
ACCESS_PARAM = "Scale_ID"
EPOCHS = 100000
LEARNING_RATE = 0.01
# SGD_MOMENTUM = 0.5

count = 0

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = torch.nn.Linear(8, 16)
        self.dense1_bn = torch.nn.BatchNorm1d(16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.dense2_bn = torch.nn.BatchNorm1d(32)
        self.fc3 = torch.nn.Linear(32, 32)
        self.dense3_bn = torch.nn.BatchNorm1d(32)
        self.fc4 = torch.nn.Linear(32, 16)
        self.dense4_bn = torch.nn.BatchNorm1d(16)
        self.fc5 = torch.nn.Linear(16, 8)
        self.dense5_bn = torch.nn.BatchNorm1d(8)
        self.fc6 = torch.nn.Linear(8, 1)

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        x = torch.nn.functional.relu(self.dense1_bn(self.fc1(x)))
        x = torch.nn.functional.dropout(x)
        x = torch.nn.functional.relu(self.dense2_bn(self.fc2(x)))
        x = torch.nn.functional.dropout(x)
        x = torch.nn.functional.relu(self.dense3_bn(self.fc3(x)))
        x = torch.nn.functional.dropout(x)
        x = torch.nn.functional.relu(self.dense4_bn(self.fc4(x)))
        x = torch.nn.functional.dropout(x)
        x = torch.nn.functional.relu(self.dense5_bn(self.fc5(x)))
        x = torch.nn.functional.dropout(x)
        x = torch.nn.functional.relu(self.fc6(x))
        # x = F.dropout(x, training=self.training)
        # loss = torch.nn.MSELoss()
        # output = loss(out_tensor, result)
        # return F.log_softmax(x, dim=1)
        return x

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
df = pd.read_csv("data/out-train.csv")

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
	for i in xrange(((df.shape[0])/BATCH_SIZE) +1):
		curr = i * BATCH_SIZE
		req = df.loc[curr:curr+BATCH_SIZE-1, :]
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
		if count>df.shape[0]:
			count=df.shape[0]
		print("Batches done: "+str(count)+"/"+str(df.shape[0]),end="\r")
	losses/=count
	print("\nIteration "+str(j+1)+"/"+str(EPOCHS)+" done")
	print("\nLoss: "+ str(losses))
	os.system("mkdir -p checkpoints")
	torch.save(model.state_dict(), "checkpoints/"+str(j))

	# print(df.shape[0])
