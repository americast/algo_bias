import numpy as np
import pandas as pd 
import pudb

# array = [1,2,6,66,3,12,11,12,39,14]
array = [0,1,2,3,4,5,6,15,31,38,53,59,64,66,108]
g = np.zeros((10,8))
f = np.zeros((10,1))
count = 0
for x in range(1,9):
	temp = np.load("out_"+str(x)+"_lrp.npy")
	temp = np.sum(temp,axis=0,keepdims=True)
	temp = temp.transpose()


	for i in range(0,14):
		for j in range(array[i],array[i+1]):
			f[i] = f[i] + temp[j] 

	print(np.min(f))	
	f = f-np.min(f)
	f = f/np.max(f)
	g[0:,[x-1]] = f

np.save("lrp_matrix", g)
df = pd.DataFrame(g)
df.to_csv("lrp_matrix.csv", index=False)