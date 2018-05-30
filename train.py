from __future__ import print_function
import torch
import argparse
import numpy as np
import pandas as pd
import sys
import os

BATCH_SIZE = 8192
EPOCHS = 100000
LEARNING_RATE = 0.001

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
        self.fc6 = torch.nn.Linear(8, 3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.nn.functional.dropout(torch.nn.functional.relu(self.fc1(x)))
        x = torch.nn.functional.relu(self.dense2_bn(self.fc2(x)))
        x = torch.nn.functional.dropout(x)
        x = torch.nn.functional.relu(self.dense3_bn(self.fc3(x)))
        x = torch.nn.functional.dropout(x)
        x = torch.nn.functional.relu(self.dense4_bn(self.fc4(x)))
        x = torch.nn.functional.dropout(x)
        x = torch.nn.functional.relu(self.dense5_bn(self.fc5(x)))
        x = torch.nn.functional.dropout(x)
        x = self.softmax(self.fc6(x))
        return x

def create_tensor(req):
	to_drop=[]
	global count
	array_now = req.loc[:,["Agency_Text","Sex_Code_Text","Ethnic_Code_Text","Language","LegalStatus","CustodyStatus","MaritalStatus","Age"]].values
	# print("array_now: ", array_now)
	array_now = torch.cuda.FloatTensor(array_now)
	array_now.cuda()

	result_now = req["DecileScore"].values
	# print("result_now: ", result_now)
	result_now =torch.cuda.LongTensor(result_now)
	result_now.cuda()
	return array_now, result_now


# Read the data

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
loss = torch.nn.CrossEntropyLoss()

df = pd.read_csv("data/out-train.csv")


cols = np.array([3,7,8,9,10,13,14,15,16,17,18,20,23])
cols += 1
df = df[df.columns[cols]]

train_flag = True

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.01, patience = 1000, verbose = True)
for j in xrange(EPOCHS):
	losses=0.0
	global count
	count = 0
	df = df.sample(frac=1).reset_index(drop=True)
	for i in xrange(((df.shape[0])/BATCH_SIZE) +1):
		if not train_flag:
			if (count>=(df.shape[0])-1):
				break 
		curr = i * BATCH_SIZE
		if train_flag:
			req = df.loc[curr:curr+BATCH_SIZE-1, :]
		else:
			req = df.loc[curr:curr+BATCH_SIZE, :]
		final_tensor, result = create_tensor(req)

		optimizer.zero_grad()
		output = model(final_tensor)
		# print(output,result)

		loss = criterion(output, result)
		loss.backward()
		optimizer.step()
		# scheduler.step(loss_here)

		losses+=loss.data[0]

		count+=BATCH_SIZE
		if count>df.shape[0]:
			count=df.shape[0]
		print("Batches done: "+str(count)+"/"+str(df.shape[0]),end="\r")
	losses/=count
	print("\nIteration "+str(j+1)+"/"+str(EPOCHS)+" done")
	print("\nLoss: "+ str(losses))
	os.system("mkdir -p checkpoints")
	torch.save(model.state_dict(), "checkpoints/"+str(j))