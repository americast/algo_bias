import numpy as np
import cv2

a = np.load("lrp_matrix.npy")
a = np.repeat(a, 8, axis = 0)
a = np.repeat(a, 8, axis = 1)
cv2.imwrite("pic_1.png", a*255)
# cv2.imshow("a", a)
# cv2.waitKey(0)