import numpy
import cv2

img = cv2.imread('test_im.jpg',0)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
