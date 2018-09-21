import numpy as np
import pandas as pd 
import pudb

g=np.zeros((8,166))
for x in range(1,9):
	f = np.load("out_"+str(x)+"_lrp.npy")
	f = np.sum(f,axis=0,keepdims=True)
	f = f+np.absolute(np.min(f))
	f = f/np.max(f)
	g[x-1] = f

np.save("lrp_matrix", g)
df = pd.DataFrame(g)
df.to_csv("lrp_matrix.csv")	