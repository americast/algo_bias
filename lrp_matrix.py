import numpy as np
import pandas as pd 
import pudb

# df = pd.read_csv("data/people.csv")
# cols = df.columns.values
# req_cols =[cols[i] for i in (4,5,7,8,9,10,11,16,24)]

# array = [1,2,6,66,3,12,11,12,39,14]
layer_num = [2,6,10,14,18,22,26,30,34,38,42, 44]
array = [0,1,3,9,75,78,90,101,113,152,166]
g = np.zeros((10,13))
f = np.zeros((10,1))
count = 0
k=1
for x in layer_num:
	temp = np.load("out_lrp_"+str(x)+".npy")
	print(temp.shape)
	temp = np.sum(temp,axis=0,keepdims=True)
	print(temp.shape)
	temp = temp.transpose()
	print(temp.shape)
	for i in range(0,10):
		for j in range(array[i],array[i+1]):
			if (j>107):
				break
			f[i] = f[i] + temp[j] 
	k=k+1
	print(np.min(f))	
	f = f-np.min(f)
	f = f/np.max(f)
	print("X: ", x)
	g[0:,[k-1]] = f
	name="lrp_matrix"+str(x)
	np.save(name, g)
