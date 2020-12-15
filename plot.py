import numpy as np
import cv2

layer_num = [2,6,10,14,18,22,26,30,34,38,42, 44]

for x in layer_num:
  a=np.load("lrp_matrix"+str(x)+".npy")
  a=np.repeat(a, 13, axis=0)
  a=np.repeat(a, 13, axis=1)
  name="pic_"+str(x)+".png"
  cv2.imwrite(name, a*255)
