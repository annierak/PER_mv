import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_image(img,size=800):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', size, size)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image = cv2.imread('test_im.jpg',cv2.IMREAD_COLOR)

p1 = (3,3)
p2 = (8,8)

cv2.line(image, p1,p2, (0,0,255), 2)
show_image(image)
