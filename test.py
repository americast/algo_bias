from __future__ import print_function
import torch
import argparse
import numpy as np
import pandas as pd
import sys
import os

BATCH_SIZE = 8192
EPOCHS = 1
LEARNING_RATE = 0.001
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
        self.fc6 = torch.nn.Linear(8, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
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
        # x = F.dropout(x, training=self.training)
        # loss = torch.nn.MSELoss()
        # output = loss(out_tensor, result)
        # return F.log_softmax(x, dim=1)
        return x

def create_tensor(req):
    to_drop=[]
    global count
    array_now = req.loc[:,["Agency_Text","Sex_Code_Text","Ethnic_Code_Text","Language","LegalStatus","CustodyStatus","MaritalStatus","Age"]].values
    array_now = torch.FloatTensor(array_now)
    array_now.cuda()

    result_now = req.loc[:,["DecileScore"]].values
    result_now =torch.FloatTensor(result_now)
    result_now.cuda()
    return array_now, result_now


parser = argparse.ArgumentParser(description='Algo_bias')
parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, metavar='N',
                        help='input batch size for training (default: 9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=EPOCHS, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=LEARNING_RATE, metavar='LR',
                    help='learning rate (default: 0.01)')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)

train_flag = False

path = raw_input("Enter path to saved model: ")
model.load_state_dict(torch.load(path))
model.eval()
df = pd.read_csv("data/out-test.csv")

cols = np.array([3,7,8,9,10,13,14,15,16,17,18,20,22])
cols += 1
df = df[df.columns[cols]]



with torch.no_grad():
    for i in xrange(df.shape[0]):
        req = df.loc[i:i+1, :]
        final_tensor, result = create_tensor(req)

        output = model(final_tensor)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == output).sum().item()

# with torch.no_grad():
#     total_iterations = len(list(testloader))
#     k = 0
#     for data in testloader:
#         k = k+1
#         images, labels = data
#         # pu.db
#         if use_gpu:
#             images = images.cuda()
#             labels = labels.cuda()
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         print("{} of {} iterations done".format(k,total_iterations))

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / df.shape[0]))
