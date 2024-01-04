import cv2
import numpy as np

image = np.zeros([2048, 2048, 3], dtype=np.uint16)
image[:,:,:] = 100
M = np.float32([[1, 0, 100], [0, 1, 40]])
moveImage = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderValue=(114, 114, 114))

cv2.namedWindow('move',cv2.WINDOW_NORMAL)
cv2.imshow('move', moveImage)
cv2.waitKey()
cv2.destroyAllWindows()
