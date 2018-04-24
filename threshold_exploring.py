import cv2
import numpy as np
import matplotlib.pyplot as plt

vidcap = cv2.VideoCapture('NPF_2.avi')
_,image = vidcap.read()
plt.figure(1)
plt.imshow(image)

thresholds = [90,95,100,105,110,115]
threshold_images = []
for threshold in thresholds:
    rval, threshold_image = cv2.threshold(image, threshold, np.iinfo(image.dtype).max, cv2.THRESH_BINARY_INV)
    threshold_images.append(threshold_image)

plt.figure(2)
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(threshold_images[i],'gray')
    plt.xticks([]),plt.yticks([])
    plt.title(str(thresholds[i]))
plt.show()
